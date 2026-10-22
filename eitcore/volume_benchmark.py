"""Scalable 3D comparison using the same CEM and explicit voxel parameters.

Finite-element fields retain the complete column/electrode geometry. Unknown
conductivity is represented on a configurable nonuniform cuboid grid; this
prevents a 256-electrode experiment from allocating a cellwise terabyte Jacobian.
The four algorithms use complex difference voltages and real conductivity.
"""
import csv
import gc
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from .physical import (EPSILON_0, material_admittivity, sample_materials,
                       normalize_config, drive_patterns)
from .physical_workflow import _solver, write_vtu, plot_spectrum


def _atomic_npz(path,**values):
    path=Path(path)
    temporary=path.with_suffix('.partial.npz')
    np.savez_compressed(temporary,**values)
    temporary.replace(path)


def _atomic_npy(path,values):
    path=Path(path)
    temporary=path.with_suffix('.partial.npy')
    np.save(temporary,values)
    temporary.replace(path)


class ParameterGrid:
    def __init__(self, solver, spec, radius):
        self.spec, self.radius = spec, radius
        self.edges = [np.linspace(-radius, radius, spec['nx']+1),
                      np.linspace(-radius, radius, spec['ny']+1),
                      np.asarray(spec['z_edges_m'], float)]
        self.shape = tuple(len(e)-1 for e in self.edges)
        points = solver.cell_centers()
        ijk = tuple(np.clip(np.searchsorted(e, points[:, d], side='right')-1,
                            0, len(e)-2) for d, e in enumerate(self.edges))
        raw = np.ravel_multi_index(ijk, self.shape)
        self.active, self.cell_parameter = np.unique(raw, return_inverse=True)
        self.n = len(self.active)
        self.P = csr_matrix((np.ones(len(points)), (np.arange(len(points)), self.cell_parameter)),
                            shape=(len(points), self.n))
        self.centres = np.stack(np.meshgrid(*[(e[:-1]+e[1:])/2 for e in self.edges],
                                           indexing='ij'), axis=-1).reshape(-1, 3)[self.active]
        index = np.full(np.prod(self.shape), -1, int)
        index[self.active] = np.arange(self.n)
        index = index.reshape(self.shape)
        widths = [np.diff(e) for e in self.edges]
        rows, cols, values = [], [], []
        row = 0
        for cell in np.ndindex(*self.shape):
            a = index[cell]
            if a < 0:
                continue
            for dim in range(3):
                other = list(cell); other[dim] += 1
                if other[dim] >= self.shape[dim]:
                    continue
                b = index[tuple(other)]
                if b < 0:
                    continue
                face_area = np.prod([widths[d][cell[d]] for d in range(3) if d != dim])
                distance = (widths[dim][cell[dim]]+widths[dim][other[dim]])/2
                weight = np.sqrt(face_area/(distance*radius))
                rows.extend([row, row]); cols.extend([a, b]); values.extend([weight, -weight])
                row += 1
        self.D = csr_matrix((values, (rows, cols)), shape=(row, self.n))
        self.volume = np.asarray(self.P.T@solver.cell_area).ravel()

    def to_cells(self, parameters):
        return np.asarray(parameters)[self.cell_parameter]

    def voxel_values(self, parameters, fill=np.nan):
        values = np.full(np.prod(self.shape), fill)
        values[self.active] = parameters
        return values.reshape(self.shape)

    def at_points(self, parameters, points):
        ijk = tuple(np.clip(np.searchsorted(e, points[:, d], side='right')-1,
                            0, len(e)-2) for d, e in enumerate(self.edges))
        values = self.voxel_values(parameters)
        result = values[ijk]
        bad = ~np.isfinite(result)
        if np.any(bad):
            result[bad] = np.asarray(parameters)[cKDTree(self.centres).query(points[bad])[1]]
        return result


class Measurements:
    def __init__(self, cfg):
        mesh, r = cfg['mesh'], cfg['reconstruction']
        n, rings = mesh['electrodes_per_ring'], mesh['n_rings']
        self.pairs = np.array([(ring*n+e, ring*n+(e+1)%n)
                              for ring in range(rings)
                              for e in range(0, n, r['measurement_stride'])], int)
        self.I = drive_patterns(cfg)
        self.mask = np.ones((len(self.I), len(self.pairs)), bool)
        if r['exclude_driven']:
            for p, (a, b) in enumerate(self.pairs):
                self.mask[:, p] = (self.I[:, a] == 0) & (self.I[:, b] == 0)
        self.unit_patterns = np.zeros((len(self.pairs), mesh['n_electrodes']))
        for row, (a, b) in zip(self.unit_patterns, self.pairs):
            row[a], row[b] = 1, -1

    def values(self, U):
        U = np.asarray(U)
        return (U[:, self.pairs[:, 0]]-U[:, self.pairs[:, 1]])[self.mask]


def compressed_jacobian(solver, sigma, grid, measurements, u=None, chunk=8, log=print):
    """Negative complex derivative of selected voltage differences by voxel.

    Every cell integral is accumulated into its actual parameter column before
    proceeding to the next drive, never forming the full electrode/cell Jacobian.
    """
    if u is None:
        u, _ = solver.forward_solve(sigma)
    adjoint, _ = solver.forward_solve(sigma, measurements.unit_patterns)
    gm = np.asarray(solver.cellwise_gradients_fast(adjoint))
    del adjoint
    rows = int(measurements.mask.sum())
    J = np.empty((rows, grid.n), dtype=complex)
    offset = 0
    for p, potential in enumerate(u):
        gu = solver.cellwise_gradients_fast([potential])[0]
        chosen = np.flatnonzero(measurements.mask[p])
        for start in range(0, len(chosen), chunk):
            indices = chosen[start:start+chunk]
            integrals = np.einsum('ecd,cd->ec', gm[indices], gu)*solver.cell_area
            block = np.asarray(grid.P.T@integrals.T).T
            J[offset:offset+len(indices)] = block
            offset += len(indices)
        if p % 30 == 0:
            log(f'    Jacobian: drive {p+1}/{len(u)}, {rows} measurements x {grid.n} parameters')
    return J


def _release(solver):
    if getattr(solver, '_cached_lu', None) is not None:
        solver._cached_lu.destroy()
        solver._cached_lu = None
    gc.collect()


def generate_truth(cfg, out, log):
    """Checkpoint each frequency and preserve the fine-mesh homogeneous frame."""
    frequencies, Uall, Ubackground = [], [], []
    for f in cfg['frequencies_hz']:
        checkpoint = out/f'forward_{f:g}Hz.npz'
        if checkpoint.exists():
            data = np.load(checkpoint)
            frequencies.append(f); Uall.append(data['U_v']); Ubackground.append(data['U_background_v'])
            log(f'Loaded completed fine-mesh forward checkpoint at {f:g} Hz')
            continue
        start = time.perf_counter()
        log(f'Fine-mesh forward simulation at {f:g} Hz')
        solver, sigma = _solver(cfg, cfg['mesh'], f)
        points = solver.cell_centers()
        values, eps = sample_materials(cfg, points, f)
        sigma.x.array[:] = values
        solver.relative_permittivity.x.array[:] = eps
        log(f'  {solver.n_cells} tetrahedra, {solver.dofs} nodes, {len(solver.Inj)} drive patterns')
        u, U = solver.forward_solve(sigma)
        U = np.asarray(U)
        write_vtu(out/f'truth_{f:g}Hz.vtu', solver,
                  {'conductivity_S_per_m': values, 'relative_permittivity': eps},
                  {'potential_real_V_pattern0': np.real(u[0]), 'potential_imag_V_pattern0': np.imag(u[0])})
        np.savez_compressed(out/f'truth_field_{f:g}Hz.npz', points_m=points,
                            conductivity_s_m=values, relative_permittivity=eps)
        del u
        bg = material_admittivity(cfg['background'], f)
        sigma.x.array[:] = bg.real
        solver.relative_permittivity.x.array[:] = bg.imag/(2*np.pi*f*EPSILON_0)
        u, Ubg = solver.forward_solve(sigma)
        del u
        Ubg = np.asarray(Ubg)
        gauge = float(np.max(np.abs(U.sum(axis=1))))
        volume = float(solver.cell_area.sum())
        sphere_volume = float(np.sum(solver.cell_area[values == 0]))
        info = {'frequency_hz': f, 'n_cells': solver.n_cells, 'n_nodes': solver.dofs,
                'elapsed_seconds': time.perf_counter()-start, 'max_gauge_v': gauge,
                'domain_volume_m3': volume, 'sphere_volume_mesh_m3': sphere_volume,
                'electrode_area_min_m2': float(solver.electrode_areas.min()),
                'electrode_area_max_m2': float(solver.electrode_areas.max()),
                'max_abs_voltage_v': float(np.abs(U).max())}
        (out/f'forward_{f:g}Hz.json').write_text(json.dumps(info, indent=2))
        _atomic_npz(checkpoint, U_v=U, U_background_v=Ubg,
                            injection_a=solver.Inj, electrode_areas_m2=solver.electrode_areas)
        shutil.copyfile(solver.mesh_name, out/'truth_mesh.msh')
        log(f'  Completed in {info["elapsed_seconds"]:.1f}s; gauge {gauge:.2e} V; saved {checkpoint}')
        frequencies.append(f); Uall.append(U); Ubackground.append(Ubg)
        _release(solver)
        del solver, sigma
    np.savez_compressed(out/'measurements.npz', version=2, frequencies_hz=frequencies,
                        U_v=Uall, U_background_v=Ubackground, injection_a=drive_patterns(cfg))
    plot_spectrum(out/'frequency_response.png', np.array(frequencies), np.array(Uall), drive_patterns(cfg))
    with (out/'measurements.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['frequency_Hz', 'pattern_index', 'electrode_1based', 'real_V_RMS', 'imag_V_RMS', 'magnitude_V_RMS', 'phase_deg'])
        for f, U in zip(frequencies, Uall):
            writer.writerows((f, p, e+1, z.real, z.imag, abs(z), np.angle(z, deg=True))
                             for p, row in enumerate(U) for e, z in enumerate(row))
    return np.asarray(Uall), np.asarray(Ubackground)


def _normal(J):
    # Real parameters, both complex voltage quadratures.
    return J.real.T@J.real+J.imag.T@J.imag


def _transpose(J, r):
    return J.real.T@r.real+J.imag.T@r.imag


def _fista(H, b, alpha, current, lower, upper, iterations):
    """Bound-constrained L1 proximal solve of a GN quadratic subproblem."""
    v = np.ones(len(b))/np.sqrt(len(b))
    for _ in range(20):
        v = H@v
        v /= max(np.linalg.norm(v), 1e-30)
    lipschitz = max(float(v@(H@v))*1.05, 1e-12)
    x, y, momentum = current.copy(), current.copy(), 1.0
    for _ in range(iterations):
        z = y-(H@y-b)/lipschitz
        new = np.clip(np.sign(z)*np.maximum(np.abs(z)-alpha/lipschitz, 0), lower, upper)
        t = (1+np.sqrt(1+4*momentum**2))/2
        y = new+(momentum-1)/t*(new-x)
        x, momentum = new, t
    return x


def inverse_algorithm(cfg, solver, sigma, grid, measurement, observed, reference,
                      algorithm, initial_J, data_scale, log):
    r = cfg['reconstruction']
    n, D = grid.n, grid.D
    bg = material_admittivity(cfg['background'], solver.frequency_hz).real
    lower, upper = r['sigma_min_s_m']/bg, r['sigma_max_s_m']/bg
    q = np.ones(n)
    DTD = (D.T@D).toarray()
    H0 = _normal(initial_J)
    trace = max(float(np.trace(H0)), 1e-30)
    prior_trace = max(float(np.trace(DTD)), 1e-30)
    if algorithm == 'tv':
        weight = r['tv_regularization']*trace*r['tv_beta']/prior_trace
    else:
        weight = r['linear_regularization' if algorithm == 'linear' else 'regularization']*trace/prior_trace
    ridge = max(trace/n*1e-7, 1e-16)
    alpha = r['l1_alpha']*max(float(np.max(np.abs(_transpose(initial_J, -observed/data_scale)))), 1e-30)
    history = []
    cache = {}

    def evaluate(v):
        sigma.x.array[:] = bg*grid.to_cells(v)
        u, U = solver.forward_solve(sigma)
        residual = (measurement.values(U)-reference-observed)/data_scale
        dq = D@v
        if algorithm == 'tv':
            penalty = weight*np.sum(np.sqrt(dq**2+r['tv_beta']**2)-r['tv_beta'])
        elif algorithm == 'l1':
            penalty = alpha*np.sum(np.abs(v-1))
        else:
            penalty = .5*weight*(dq@dq)
        penalty += .5*ridge*np.sum((v-1)**2)
        return .5*np.vdot(residual, residual).real+penalty, residual, u, np.asarray(U)

    initial_objective, residual, u, U = evaluate(q)
    objective = initial_objective
    max_outer = 1 if algorithm == 'linear' else r['iterations']
    for iteration in range(max_outer):
        start = time.perf_counter()
        J = initial_J if iteration == 0 else compressed_jacobian(
            solver, sigma, grid, measurement, u, r['jacobian_chunk'], log)*bg/data_scale
        H = H0.copy() if iteration == 0 else _normal(J)
        b = _transpose(J, residual)
        if algorithm == 'tv':
            dq = D@q
            weights = 1/np.sqrt(dq**2+r['tv_beta']**2)
            prior = (D.T@D.multiply(weights[:, None])).toarray()
            H += weight*prior
            b -= weight*(D.T@(weights*dq))
            H.flat[::n+1] += ridge
            b -= ridge*(q-1)
            step = cho_solve(cho_factor(H, check_finite=False), b, check_finite=False)
        elif algorithm == 'l1':
            H.flat[::n+1] += ridge
            target = b+H@(q-1)
            candidate = _fista(H, target, alpha, q-1, lower-1, upper-1, r['l1_iterations'])
            step = candidate-(q-1)
        else:
            H += weight*DTD
            H.flat[::n+1] += ridge
            b -= weight*(DTD@q)+ridge*(q-1)
            step = cho_solve(cho_factor(H, check_finite=False), b, check_finite=False)
        accepted, length = False, 1.0
        for _ in range(9):
            trial = np.clip(q+length*step, lower, upper)
            new_objective, new_residual, new_u, new_U = evaluate(trial)
            if new_objective < objective:
                q, objective, residual, u, U = trial, new_objective, new_residual, new_u, new_U
                accepted = True
                break
            length /= 2
        item = {'iteration': iteration+1, 'objective': float(objective),
                'relative_difference_voltage_residual': float(np.linalg.norm(residual)),
                'step_length': length if accepted else 0,
                'elapsed_seconds': time.perf_counter()-start}
        history.append(item)
        log(f'  {algorithm} {iteration+1}/{max_outer}: residual={item["relative_difference_voltage_residual"]:.5g}, objective={objective:.5g}, step={item["step_length"]:g}')
        del H
        if not accepted:
            break
    objective, residual, u, U = evaluate(q)
    return bg*q, history, {'initial_objective': float(initial_objective), 'final_objective': float(objective),
                           'relative_difference_voltage_residual': float(np.linalg.norm(residual)),
                           'effective_prior_weight': float(weight) if algorithm!='l1' else 0.0,
                           'effective_l1_alpha': float(alpha) if algorithm=='l1' else 0.0}


def run(config, log=print):
    cfg = normalize_config(config)
    r = cfg['reconstruction']
    out = Path(r['output_directory'] or Path(cfg['output_root'])/cfg['name'])
    out.mkdir(parents=True, exist_ok=True)
    config_path = out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError(f'{out} already contains another configuration; select a new output_directory')
    config_path.write_text(json.dumps(cfg, indent=2))
    log(f'Benchmark folder: {out.resolve()}')
    start = time.perf_counter()
    Uall, Ubg = generate_truth(cfg, out, log)
    frequency_diagnostics(cfg,out,Uall,Ubg)
    if not r['enabled']:
        save_report(out, cfg, [], time.perf_counter()-start)
        log(f'Results: {out.resolve()}')
        return out
    measurement = Measurements(cfg)
    np.savez_compressed(out/'measurement_selection.npz', pairs_0based=measurement.pairs,
                        mask=measurement.mask, injection_a=measurement.I)
    log(f'Inversion uses {measurement.mask.sum()} passive voltage differences from {len(measurement.I)} drives')
    all_metrics = []
    for f in r['frequencies_hz']:
        indices = cfg['frequencies_hz'].index(f)
        observed = measurement.values(Uall[indices]-Ubg[indices])
        rng = np.random.default_rng(r['seed'])
        observed += r['noise_std_v']*(rng.normal(size=observed.shape)+1j*rng.normal(size=observed.shape))
        data_scale = max(float(np.linalg.norm(observed)), 1e-30)
        mesh = dict(cfg['mesh'])
        mesh.update({key: r[key] for key in ('h_center', 'h_edge', 'electrode_mesh_size', 'h_band') if r.get(key) is not None})
        solver, sigma = _solver(cfg, mesh, f)
        grid = ParameterGrid(solver, r['parameter_grid'], cfg['mesh']['radius'])
        bg = material_admittivity(cfg['background'], f).real
        sigma.x.array[:] = bg
        u0, U0 = solver.forward_solve(sigma)
        reference = measurement.values(U0)
        log(f'Inverse mesh at {f:g} Hz: {solver.n_cells} cells, {grid.n} parameters')
        jac_path = out/f'background_jacobian_{f:g}Hz.npy'
        if jac_path.exists():
            J0 = np.load(jac_path)
            if J0.shape != (measurement.mask.sum(),grid.n):
                raise ValueError('Cached Jacobian shape does not match this configuration')
        else:
            J0 = compressed_jacobian(solver, sigma, grid, measurement, u0, r['jacobian_chunk'], log)*bg/data_scale
            _atomic_npy(jac_path, J0)
        del u0
        shutil.copyfile(solver.mesh_name, out/'reconstruction_mesh.msh')
        for algorithm in r['algorithms']:
            folder = out/'reconstructions'/f'{f:g}Hz'/algorithm
            folder.mkdir(parents=True, exist_ok=True)
            metrics_path = folder/'metrics.json'
            if metrics_path.exists():
                all_metrics.append(json.loads(metrics_path.read_text()))
                log(f'Loaded completed {algorithm} at {f:g} Hz')
                continue
            if algorithm == 'dbar':
                info = {'algorithm': algorithm, 'frequency_hz': f, 'status': 'unsupported',
                        'reason': 'The existing Nachman D-bar algorithm is two-dimensional and cannot reconstruct a 3D volume.'}
                metrics_path.write_text(json.dumps(info, indent=2))
                all_metrics.append(info)
                continue
            log(f'Running {algorithm} at {f:g} Hz')
            t = time.perf_counter()
            parameters, history, fit = inverse_algorithm(cfg, solver, sigma, grid, measurement,
                                                         observed, reference, algorithm, J0, data_scale, log)
            rec = grid.to_cells(parameters)
            points = solver.cell_centers()
            truth, eps_truth = sample_materials(cfg, points, f)
            np.savez_compressed(folder/'reconstruction.npz', points_m=points,
                                conductivity_s_m=rec, parameter_centres_m=grid.centres,
                                parameter_conductivity_s_m=parameters,
                                frequency_hz=f, observed_difference_v=observed)
            write_vtu(folder/'reconstruction.vtu', solver,
                      {'conductivity_S_per_m': rec, 'truth_conductivity_S_per_m': truth,
                       'contrast_S_per_m': rec-bg, 'relative_permittivity_fixed': solver.relative_permittivity.x.array.real})
            write_legacy_vtk(folder/'reconstruction.vtk', solver, rec, truth)
            metrics = evaluate_reconstruction(cfg, grid, parameters, f)
            metrics.update(fit, algorithm=algorithm, frequency_hz=f, status='complete',
                           elapsed_seconds=time.perf_counter()-t, n_parameters=grid.n,
                           inverse_cells=solver.n_cells, history=history,
                           implementation='3D voxel parameterization; complex difference voltages; fixed background permittivity')
            plot_volume(folder/'reconstruction_3d.png', cfg, grid, parameters, f, algorithm)
            plot_sections(folder/'reconstruction_sections.png', cfg, grid, parameters, f, algorithm)
            metrics_path.write_text(json.dumps(metrics, indent=2))
            all_metrics.append(metrics)
            log(f'Saved {folder}; contrast L2 error={metrics["relative_L2_contrast"]:.4g}, Dice={metrics["dice_half_peak"]:.3f}')
        _release(solver)
        del solver, sigma, J0, grid
        gc.collect()
        save_report(out, cfg, all_metrics, time.perf_counter()-start)
    if r['refinement_check']:
        refinement_check(cfg,out,log)
    save_report(out, cfg, all_metrics, time.perf_counter()-start)
    log(f'Results: {out.resolve()}')
    return out


def frequency_diagnostics(cfg,out,Uall=None,Ubg=None):
    """Observed spectrum, signal strength and required source voltage in SI."""
    if Uall is None:
        data=np.load(out/'measurements.npz')
        Uall,Ubg=data['U_v'],data['U_background_v']
    measurements=Measurements(cfg)
    signals=[measurements.values(U-B) for U,B in zip(Uall,Ubg)]
    base=signals[0]
    rows=[]
    I=drive_patterns(cfg)
    source,sink=np.argmax(I,axis=1),np.argmin(I,axis=1)
    gains=[]
    for f,U,signal in zip(cfg['frequencies_hz'],Uall,signals):
        gamma=material_admittivity(cfg['background'],f)
        gain=np.vdot(base,signal)/max(np.vdot(base,base).real,1e-30)
        gains.append(gain)
        rows.append({'frequency_hz':f,'water_admittivity_real_S_m':gamma.real,
                     'water_admittivity_imag_S_m':gamma.imag,
                     'water_admittivity_phase_deg':float(np.angle(gamma,deg=True)),
                     'object_signal_rms_v':float(np.sqrt(np.mean(np.abs(signal)**2))),
                     'object_signal_max_v':float(np.max(np.abs(signal))),
                     'object_signal_gain_relative_to_first_magnitude':float(abs(gain)),
                     'object_signal_gain_relative_to_first_phase_deg':float(np.angle(gain,deg=True)),
                     'max_source_pair_voltage_v_rms':float(np.max(np.abs(U[np.arange(len(I)),source]-U[np.arange(len(I)),sink]))),
                     'min_real_input_power_w':float(np.min(np.sum(I*np.conj(U),axis=1).real))})
    (out/'frequency_diagnostics.json').write_text(json.dumps(rows,indent=2))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(7,6),constrained_layout=True)
    f=np.asarray(cfg['frequencies_hz']);gains=np.asarray(gains)
    axes[0].plot(f,np.abs(gains),'o-');axes[0].set_ylabel('Object-signal gain magnitude [1]')
    axes[1].plot(f,np.angle(gains,deg=True),'o-');axes[1].set_ylabel('Object-signal relative phase [degree]')
    for ax in axes:
        if np.all(f>0):ax.set_xscale('log')
        ax.set_xlabel('Frequency [Hz]');ax.grid(alpha=.3)
    fig.suptitle('Measured frequency effect, normalized to the first frequency')
    fig.savefig(out/'frequency_effects.png',dpi=150);plt.close(fig)
    return rows


def refinement_check(cfg,out,log=print):
    """Independent two-level forward mesh study on a representative drive subset."""
    path=out/'mesh_refinement.json'
    if path.exists():
        return json.loads(path.read_text())
    r=cfg['reconstruction']
    frequency=cfg['frequencies_hz'][0]
    base=np.load(out/f'forward_{frequency:g}Hz.npz')
    m=dict(cfg['mesh'])
    for key in ('h_center','h_edge','h_band','electrode_mesh_size'):
        if m.get(key) is not None:
            m[key]*=r['refinement_factor']
    log('Building independently refined mesh for a two-level forward comparison ...')
    solver,sigma=_solver(cfg,m,frequency)
    selected=np.unique(np.linspace(0,len(solver.Inj)-1,min(r['refinement_drives'],len(solver.Inj))).round().astype(int))
    currents=solver.Inj[selected]
    values,eps=sample_materials(cfg,solver.cell_centers(),frequency)
    sigma.x.array[:]=values; solver.relative_permittivity.x.array[:]=eps
    u,U=solver.forward_solve(sigma,currents)
    U=np.asarray(U)
    predicted=solver.M[solver.dofs:solver.dofs+solver.L,:solver.dofs]@np.asarray(u).T
    predicted+=solver.M[solver.dofs:solver.dofs+solver.L,solver.dofs:solver.dofs+solver.L]@U.T
    current_error=float(np.linalg.norm(predicted.T-currents)/np.linalg.norm(currents))
    del u
    bg=material_admittivity(cfg['background'],frequency)
    sigma.x.array[:]=bg.real
    solver.relative_permittivity.x.array[:]=bg.imag/(2*np.pi*frequency*EPSILON_0)
    u,Ubg=solver.forward_solve(sigma,currents);del u
    Ubg=np.asarray(Ubg)
    measurements=Measurements(cfg)
    a,b=measurements.pairs.T
    mask=measurements.mask[selected]
    coarse=((base['U_v'][selected]-base['U_background_v'][selected])[:,a]-
            (base['U_v'][selected]-base['U_background_v'][selected])[:,b])[mask]
    fine=((U-Ubg)[:,a]-(U-Ubg)[:,b])[mask]
    result={'frequency_hz':frequency,'selected_drive_indices':selected.tolist(),
            'refinement_factor':r['refinement_factor'],'refined_cells':solver.n_cells,
            'refined_nodes':solver.dofs,'object_signal_relative_change':float(np.linalg.norm(fine-coarse)/np.linalg.norm(fine)),
            'passive_voltage_difference_relative_change':float(np.linalg.norm((U[:,a]-U[:,b])[mask]-(base['U_v'][selected][:,a]-base['U_v'][selected][:,b])[mask])/np.linalg.norm((U[:,a]-U[:,b])[mask])),
            'relative_electrode_current_error':current_error,
            'refined_electrode_area_min_m2':float(solver.electrode_areas.min()),
            'refined_electrode_area_max_m2':float(solver.electrode_areas.max()),
            'sphere_volume_refined_m3':float(np.sum(solver.cell_area[values==0])),
            'sphere_volume_exact_m3':float(4/3*np.pi*cfg['inclusions'][0]['radius_m']**3),
            'interpretation':'Two mesh levels quantify discretization sensitivity; they do not alone establish asymptotic convergence.'}
    path.write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'mesh_refinement_measurements.npz',frequency_hz=frequency,
                        drive_indices=selected,U_refined_v=U,U_background_refined_v=Ubg,
                        difference_fine_v=fine,difference_original_v=coarse)
    shutil.copyfile(solver.mesh_name,out/'refined_truth_mesh.msh')
    write_vtu(out/'refined_truth.vtu',solver,{'conductivity_S_per_m':values,'relative_permittivity':eps})
    log(f'Refinement check: {solver.n_cells} cells, object-voltage change {result["object_signal_relative_change"]:.3%}')
    _release(solver)
    return result


def evaluation_grid(cfg):
    R, H = cfg['mesh']['radius'], cfg['mesh']['height']
    x = np.linspace(-R, R, 61)
    z = np.linspace(0, H, 251)
    p = np.stack(np.meshgrid(x, x, z, indexing='ij'), axis=-1).reshape(-1, 3)
    return p[np.sum(p[:, :2]**2, axis=1) <= R**2]


def evaluate_reconstruction(cfg, grid, parameters, f):
    p = evaluation_grid(cfg)
    truth, _ = sample_materials(cfg, p, f)
    rec = grid.at_points(parameters, p)
    bg = material_admittivity(cfg['background'], f).real
    dt, dr = truth-bg, rec-bg
    true_mask = np.abs(dt) > bg*.05
    detected = np.abs(dr) > .5*max(np.abs(dr).max(), 1e-30)
    dice = 2*np.sum(true_mask & detected)/max(np.sum(true_mask)+np.sum(detected), 1)
    weight = np.maximum(bg-rec, 0)
    centre = np.sum(p*weight[:, None], axis=0)/max(weight.sum(), 1e-30)
    true_centre = np.asarray(cfg['inclusions'][0]['center_m'])
    return {'relative_L2_conductivity': float(np.linalg.norm(rec-truth)/np.linalg.norm(truth)),
            'relative_L2_contrast': float(np.linalg.norm(dr-dt)/np.linalg.norm(dt)),
            'rmse_S_per_m': float(np.sqrt(np.mean((rec-truth)**2))),
            'dice_half_peak': float(dice), 'centroid_m': centre.tolist(),
            'centroid_error_m': float(np.linalg.norm(centre-true_centre)),
            'minimum_conductivity_S_per_m': float(rec.min()),
            'maximum_conductivity_S_per_m': float(rec.max()),
            'truth_sphere_mean_reconstructed_S_per_m': float(rec[true_mask].mean()),
            'recovered_mean_sphere_contrast_fraction': float((bg-rec[true_mask].mean())/bg)}


def write_legacy_vtk(path, solver, rec, truth):
    xyz = solver.omega.geometry.x
    cells = np.asarray(solver.omega.geometry.dofmap).reshape(-1, 4)
    dg = np.array([solver.V_sigma.dofmap.cell_dofs(c)[0] for c in range(len(cells))])
    with Path(path).open('w') as fh:
        fh.write('# vtk DataFile Version 3.0\n3D EIT conductivity [S/m]\nASCII\nDATASET UNSTRUCTURED_GRID\n')
        fh.write(f'POINTS {len(xyz)} double\n')
        np.savetxt(fh, xyz, fmt='%.12g')
        fh.write(f'CELLS {len(cells)} {5*len(cells)}\n')
        np.savetxt(fh, np.column_stack([np.full(len(cells), 4), cells]), fmt='%d')
        fh.write(f'CELL_TYPES {len(cells)}\n')
        np.savetxt(fh, np.full(len(cells), 10), fmt='%d')
        fh.write(f'CELL_DATA {len(cells)}\n')
        for name, values in [('conductivity_S_per_m', rec), ('truth_conductivity_S_per_m', truth)]:
            fh.write(f'SCALARS {name} double 1\nLOOKUP_TABLE default\n')
            np.savetxt(fh, np.asarray(values)[dg], fmt='%.12g')


def plot_volume(path, cfg, grid, parameters, f, algorithm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    bg = material_admittivity(cfg['background'], f).real
    values = grid.voxel_values(parameters)
    contrast = bg-values
    cutoff = .5*np.nanmax(np.abs(contrast))
    occupied = np.isfinite(values) & (np.abs(contrast) >= max(cutoff, 1e-12))
    vertices = np.meshgrid(*grid.edges, indexing='ij')
    norm = Normalize(0, bg)
    colors = plt.get_cmap('viridis')(norm(np.nan_to_num(values, nan=bg)))
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    R = cfg['mesh']['radius']
    from mesh_tools import ring_heights
    heights=ring_heights(cfg['mesh'])
    zlo, zhi = max(0, heights[0]-.025), min(cfg['mesh']['height'], heights[-1]+.025)
    for j, title in enumerate(['Truth: insulating sphere', f'{algorithm}: half-peak contrast support']):
        ax = fig.add_subplot(1, 2, j+1, projection='3d')
        theta = np.linspace(0, 2*np.pi, 80)
        for z in heights:
            ax.plot(R*np.cos(theta), R*np.sin(theta), z, color='gray', alpha=.35, linewidth=.6)
        if j == 0:
            inc = cfg['inclusions'][0]
            a, b = np.meshgrid(np.linspace(0, 2*np.pi, 64), np.linspace(0, np.pi, 32))
            c, rad = inc['center_m'], inc['radius_m']
            ax.plot_surface(c[0]+rad*np.cos(a)*np.sin(b), c[1]+rad*np.sin(a)*np.sin(b),
                            c[2]+rad*np.cos(b), color=plt.get_cmap('viridis')(0), linewidth=0)
        else:
            ax.voxels(*vertices, occupied, facecolors=colors, edgecolor=(0,0,0,.08), linewidth=.2)
        ax.set(xlim=(-R,R), ylim=(-R,R), zlim=(zlo,zhi), xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title=title)
        ax.set_box_aspect((2*R, 2*R, zhi-zlo))
        ax.view_init(elev=17, azim=38)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=fig.axes, shrink=.6,
                 label='Conductivity [S/m] — common truth/reconstruction scale')
    fig.suptitle(f'{algorithm}, {f:g} Hz — instrumented region; full column is present in VTK')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sections(path, cfg, grid, parameters, f, algorithm):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    R, H = cfg['mesh']['radius'], cfg['mesh']['height']
    centre = np.asarray(cfg['inclusions'][0]['center_m'])
    bg = material_admittivity(cfg['background'], f).real
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for col, (a,b) in enumerate([(0,1),(0,2),(1,2)]):
        x = np.linspace(-R,R,180)
        band=cfg['mesh'].get('refinement_z') or [0,H]
        y = np.linspace(*band,220) if b==2 else x
        X,Y=np.meshgrid(x,y)
        p=np.tile(centre,(X.size,1)); p[:,a]=X.ravel(); p[:,b]=Y.ravel()
        truth,_=sample_materials(cfg,p,f)
        rec=grid.at_points(parameters,p)
        keep=np.sum(p[:,:2]**2,axis=1)<=R**2
        for row, values in enumerate((truth,rec)):
            values[~keep]=np.nan
            ax=axes[row,col]
            im=ax.imshow(values.reshape(X.shape),origin='lower',extent=(x[0],x[-1],y[0],y[-1]),
                         aspect='equal',cmap='viridis',norm=Normalize(0,bg))
            ax.set(xlabel=f'{"xyz"[a]} [m]',ylabel=f'{"xyz"[b]} [m]',
                   title=f'{"Truth" if row==0 else algorithm}: {"xyz"[a]}{"xyz"[b]} section')
    fig.colorbar(im,ax=axes,label='Conductivity [S/m]',shrink=.8)
    fig.suptitle(f'{algorithm}, {f:g} Hz — sections through the known truth centre (evaluation only)')
    fig.savefig(path,dpi=150); plt.close(fig)


def save_report(out, cfg, metrics, elapsed):
    (out/'comparison.json').write_text(json.dumps({'config':cfg,'elapsed_seconds_this_invocation':elapsed,
                                                  'reconstructions':metrics},indent=2))
    completed=[m for m in metrics if m['status']=='complete']
    keys=['frequency_hz','algorithm','relative_L2_contrast','dice_half_peak','centroid_error_m',
          'recovered_mean_sphere_contrast_fraction','relative_difference_voltage_residual','elapsed_seconds']
    with (out/'comparison.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(completed)
    first=cfg['inclusions'][0]['material']
    lines=[f'# {2*cfg["mesh"]["radius"]*1000:g} mm column: {cfg["mesh"]["n_electrodes"]}-electrode 3D comparison','',
           f'All configuration values are in config.json; geometry is in metres. The sphere has DC conductivity {first["conductivity_s_m"]:g} S/m and relative permittivity {first.get("relative_permittivity",0):g}.',
           f'Background conductivity: {cfg["background"]["conductivity_s_m"]:g} S/m; relative permittivity {cfg["background"].get("relative_permittivity",0):g}. The 78.3 water example uses https://nvlpubs.nist.gov/nistpubs/jres/56/jresv56n1p1_a1b.pdf as a permittivity reference.',
           '','The 3D variants use a common voxel parameterization, complex difference voltages and fixed background permittivity. GN uses a quadratic spatial prior, TV uses smoothed total variation, linear is a single background linearization, and L1 uses proximal Gauss–Newton with soft thresholding. These extend the algorithm families to 3D; they are not the original 2D CG1 L1 implementation.',
           '','D-bar is unsupported in 3D and is recorded as such at each frequency. It is never replaced by a 2D slice labelled as a volume.',
           '','All 256 electrodes are in both meshes. The configured drive stride and measurement stride explicitly select the simulated protocol. Driven-electrode measurements are excluded. The homogeneous fine-mesh reference is subtracted from the fine truth measurements; the inverse uses its own homogeneous reference.',
           '','The inverse grid covers the full 0.5 m column, with finer axial cells around the rings. No sphere location is supplied to inversion. The known sphere centre is used only for evaluation and slice plotting.',
           '',f'Current and contact resistance are assumptions, not calibrated values. Configured I/Q noise standard deviation: {cfg["reconstruction"]["noise_std_v"]:g} V. Mesh refinement and validation artifacts are reported separately when run.',
           '','Each completed algorithm/frequency folder includes reconstruction.vtk, reconstruction.vtu, a 3D image, section images, numerical arrays and metrics. VTK contains the complete column, not just the plotted sensor band.',
           '','| Frequency [Hz] | Algorithm | Contrast L2 error | Dice | Centroid error [mm] | Recovered mean contrast | Voltage residual |',
           '|---:|---|---:|---:|---:|---:|---:|']
    for m in completed:
        lines.append(f'| {m["frequency_hz"]:g} | {m["algorithm"]} | {m["relative_L2_contrast"]:.3f} | {m["dice_half_peak"]:.3f} | {m["centroid_error_m"]*1000:.2f} | {m["recovered_mean_sphere_contrast_fraction"]:.3f} | {m["relative_difference_voltage_residual"]:.4f} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
