#!/usr/bin/env python
"""Inspect and package a completed JSON-configured column experiment.

Runs without FEniCSx. Optional --render-vtk uses PyVista/VTK to read the actual
exported files and render their thresholded cell fields without interpolation.
"""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
from eitcore.physical import sample_materials, material_admittivity
from eitcore.volume_benchmark import save_report


def geometry_image(cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mesh_tools import electrode_centres
    R,H=cfg['mesh']['radius'],cfg['mesh']['height']
    fig=plt.figure(figsize=(8,10),constrained_layout=True)
    ax=fig.add_subplot(projection='3d')
    t=np.linspace(0,2*np.pi,90)
    for z in [0,H]+cfg['mesh']['ring_z']:
        ax.plot(R*np.cos(t),R*np.sin(t),z,color='gray',linewidth=.5)
    for angle in np.linspace(0,2*np.pi,9)[:-1]:
        ax.plot([R*np.cos(angle)]*2,[R*np.sin(angle)]*2,[0,H],color='gray',alpha=.25)
    radius=cfg['mesh']['electrode_diameter']/2
    for x,y,z in electrode_centres(cfg['mesh']):
        a=np.arctan2(y,x)+np.arcsin(radius*np.cos(t)/R)
        ax.plot(R*np.cos(a),R*np.sin(a),z+radius*np.sin(t),color='tab:orange',linewidth=.7)
    inc=cfg['inclusions'][0];c=inc['center_m'];r=inc['radius_m']
    a,b=np.meshgrid(np.linspace(0,2*np.pi,50),np.linspace(0,np.pi,25))
    ax.plot_surface(c[0]+r*np.cos(a)*np.sin(b),c[1]+r*np.sin(a)*np.sin(b),c[2]+r*np.cos(b),color='navy')
    ax.set(xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',zlim=(0,H),
           title=f'Full {2*R*1000:g} mm diameter x {H:g} m column\n{cfg["mesh"]["n_rings"]} rings, {cfg["mesh"]["n_electrodes"]} electrodes; insulating sphere')
    ax.set_box_aspect((2*R,2*R,H));ax.view_init(elev=9,azim=35)
    ax.set_xticks([-R,0,R],labels=[f'-{R:g}','0',f'{R:g}'])
    ax.set_yticks([-R,0,R],labels=[f'-{R:g}','0',f'{R:g}'])
    ax.tick_params(axis='x',labelsize=8,pad=1)
    ax.tick_params(axis='y',labelsize=8,pad=1)
    fig.savefig(out/'geometry_full_column.png',dpi=160);plt.close(fig)


def truth_sections(cfg,out):
    import matplotlib.pyplot as plt
    f=cfg['frequencies_hz'][0];R=cfg['mesh']['radius'];centre=np.asarray(cfg['inclusions'][0]['center_m'])
    fig,axes=plt.subplots(1,3,figsize=(12,5),constrained_layout=True)
    for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
        x=np.linspace(-R,R,180)
        y=np.linspace(*cfg['mesh']['refinement_z'],220) if b==2 else x
        X,Y=np.meshgrid(x,y);p=np.tile(centre,(X.size,1));p[:,a]=X.ravel();p[:,b]=Y.ravel()
        sigma,_=sample_materials(cfg,p,f);sigma[np.sum(p[:,:2]**2,axis=1)>R**2]=np.nan
        im=ax.imshow(sigma.reshape(X.shape),origin='lower',extent=(x[0],x[-1],y[0],y[-1]),
                     cmap='viridis',vmin=0,vmax=cfg['background']['conductivity_s_m'],aspect='equal')
        ax.set(xlabel=f'{"xyz"[a]} [m]',ylabel=f'{"xyz"[b]} [m]',title=f'Truth {"xyz"[a]}{"xyz"[b]}')
    fig.colorbar(im,ax=axes,label='Conductivity [S/m]',shrink=.7)
    fig.savefig(out/'truth_sections.png',dpi=150);plt.close(fig)


def inspect_vtk(folder,cfg,render=False):
    import pyvista as pv
    legacy=pv.read(folder/'reconstruction.vtk')
    xml=pv.read(folder/'reconstruction.vtu')
    assert legacy.n_points==xml.n_points and legacy.n_cells==xml.n_cells
    assert np.allclose(legacy.bounds,xml.bounds,rtol=1e-9,atol=1e-10)
    assert np.isclose(xml.bounds[4],0) and np.isclose(xml.bounds[5],cfg['mesh']['height'])
    assert np.all(xml.celltypes==10),'Output must contain actual 3D tetrahedra'
    values=xml.cell_data['conductivity_S_per_m']
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(legacy.cell_data['conductivity_S_per_m'],values,rtol=1e-9)
    if render:
        bg=cfg['background']['conductivity_s_m']
        xml.cell_data['absolute_contrast']=np.abs(values-bg)
        support=xml.threshold(float(np.max(np.abs(values-bg)))/2,scalars='absolute_contrast')
        plot=pv.Plotter(off_screen=True,window_size=(1200,1000))
        plot.set_background('white')
        plot.add_mesh(support,scalars='conductivity_S_per_m',clim=(0,bg),cmap='viridis',
                      show_edges=False,scalar_bar_args={'title':'Conductivity\n[S/m]',
                          'vertical':True,'position_x':.87,'position_y':.18,'width':.06,'height':.60,
                          'fmt':'%.3f','n_labels':6,'title_font_size':12,'label_font_size':11,'color':'black'})
        R=cfg['mesh']['radius'];heights=cfg['mesh']['ring_z']
        for z in heights:
            t=np.linspace(0,2*np.pi,100)
            line=pv.lines_from_points(np.column_stack([R*np.cos(t),R*np.sin(t),np.full(len(t),z)]),close=True)
            plot.add_mesh(line,color='gray',opacity=.5)
        plot.show_grid(xtitle='x [m]',ytitle='y [m]',ztitle='z [m]',color='black',
                       fmt='%.3f',font_size=10,n_xlabels=3,n_ylabels=3,n_zlabels=6)
        plot.add_text(f'{folder.name}, {folder.parent.name}\nActual VTK cell field; half-peak support\nSensor-band view',font_size=12,color='black')
        mid=(heights[0]+heights[-1])/2
        plot.camera_position=[(.18,.18,mid+.10),(0,0,mid),(0,0,1)]
        plot.camera.zoom(.80)
        plot.show(screenshot=str(folder/'vtk_field_3d.png'),auto_close=True)
    return {'reader':'PyVista/VTK '+pv.__version__,'n_points':xml.n_points,'n_tetrahedra':xml.n_cells,
            'bounds_m':list(xml.bounds),'legacy_xml_fields_match':True,'finite_conductivity':True}


def finalize(out,allow_partial=False,render=False,package=False):
    out=Path(out);cfg=json.loads((out/'config.json').read_text())
    metrics=[]
    for path in (out/'reconstructions').glob('*Hz/*/metrics.json'):
        metric=json.loads(path.read_text())
        if metric['status']=='complete':
            if metric['algorithm']=='l1':metric['effective_prior_weight']=0.0
            else:metric['effective_l1_alpha']=0.0
            path.write_text(json.dumps(metric,indent=2))
        metrics.append(metric)
    order={name:i for i,name in enumerate(cfg['reconstruction']['algorithms'])}
    metrics.sort(key=lambda m:(m['frequency_hz'],order[m['algorithm']]))
    expected=sum(a!='dbar' for a in cfg['reconstruction']['algorithms'])*len(cfg['reconstruction']['frequencies_hz'])
    completed=[m for m in metrics if m['status']=='complete']
    if not allow_partial and len(completed)!=expected:
        raise RuntimeError(f'Only {len(completed)}/{expected} reconstructions completed')
    geometry_image(cfg,out);truth_sections(cfg,out)
    validation={}
    for m in completed:
        folder=out/'reconstructions'/f'{m["frequency_hz"]:g}Hz'/m['algorithm']
        for name in ('reconstruction.vtk','reconstruction.vtu','reconstruction.npz','reconstruction_3d.png','reconstruction_sections.png'):
            assert (folder/name).stat().st_size>1000,f'Missing/empty {folder/name}'
        validation[str(folder.relative_to(out))]=inspect_vtk(folder,cfg,render)
    (out/'vtk_validation.json').write_text(json.dumps(validation,indent=2))
    forward_seconds=sum(json.loads(p.read_text())['elapsed_seconds'] for p in out.glob('forward_*Hz.json'))
    elapsed=forward_seconds+sum(m['elapsed_seconds'] for m in completed)
    save_report(out,cfg,metrics,elapsed)
    lines=['','## Numerical validation','',
           f'{len(completed)}/{expected} requested 3D reconstructions completed. Actual legacy VTK and XML VTK files were read independently using PyVista/VTK; fields, tetrahedron counts and complete-column bounds agree.',
           '','Validation logs: physical_tests.log (7 tests), scalable_tests.log (3 tests), gui_config_tests.log (4 tests), framework_tests.log (115 checks), dbar_tests.log (57 checks).',
           '','The configured L1 inner budget is 1,500 iterations per nonlinear step. The earlier 150-iteration result is retained in l1_convergence_check_150_iterations/.',
           '','## Mesh sensitivity','']
    if (out/'mesh_refinement.json').exists():
        refinement=json.loads((out/'mesh_refinement.json').read_text())
        lines.append(f'The original 258,100-cell fine mesh was compared with {refinement["refined_cells"]:,} tetrahedra. On 16 representative drives the object-induced passive voltage signal changed by {refinement["object_signal_relative_change"]:.2%}; absolute passive voltage differences changed by {refinement["passive_voltage_difference_relative_change"]:.2%}. The relative electrode-current conservation error was {refinement["relative_electrode_current_error"]:.2e}. This supports a difference-imaging reference comparison, not an assumption of zero discretization error.')
    if (out/'frequency_diagnostics.json').exists():
        diagnostics=json.loads((out/'frequency_diagnostics.json').read_text())
        d=diagnostics[-1]
        lines.extend(['','## Physical signal scale','',f'The first-frequency object signal is {diagnostics[0]["object_signal_rms_v"]*1e6:.2f} microvolt RMS across the selected channels. At {d["frequency_hz"]:g} Hz its weighted gain relative to the first frequency is {d["object_signal_gain_relative_to_first_magnitude"]:.8f}, with phase change {d["object_signal_gain_relative_to_first_phase_deg"]:.3f} degrees. Reconstructions should therefore be similar across frequency for this water/sphere model.',
                      '',f'The assumed 1 mA current and contact model require up to {max(x["max_source_pair_voltage_v_rms"] for x in diagnostics):.2f} V RMS between source electrodes. These assumptions need checking against the actual instrument and measured contacts.'])
    old=out/'l1_convergence_check_150_iterations/l1_1000Hz/metrics.json'
    newer=out/'reconstructions/1000Hz/l1/metrics.json'
    if old.exists() and newer.exists():
        study={'150_iterations':json.loads(old.read_text()),'1500_iterations':json.loads(newer.read_text())}
        (out/'l1_iteration_study.json').write_text(json.dumps(study,indent=2))
        lines.extend(['','## L1 iteration check','',f'Increasing the inner iteration budget from 150 to 1,500 improved 1 kHz Dice from {study["150_iterations"]["dice_half_peak"]:.3f} to {study["1500_iterations"]["dice_half_peak"]:.3f}, and reduced the relative difference-voltage residual from {study["150_iterations"]["relative_difference_voltage_residual"]:.3f} to {study["1500_iterations"]["relative_difference_voltage_residual"]:.3f}.'])
    lines.extend(['','## Reading the result','',
                  'TV and L1 recover sharper supports than the quadratic-prior methods for these settings. All methods still underestimate the insulating contrast. Neither these noiseless synthetic results nor the chosen regularization values establish calibrated experimental accuracy.',
                  '','Open geometry_full_column.png for the full apparatus, frequency_effects.png for the frequency sweep, or any reconstructions/<frequency>Hz/<algorithm>/reconstruction.vtk in ParaView. vtk_field_3d.png is rendered from the actual exported tetrahedral cell field, without smoothing its conductivity values.'])
    with (out/'REPORT.md').open('a') as fh:fh.write('\n'.join(lines)+'\n')
    if package:
        archive=out.parent/'column_results.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in out.rglob('*'):
                if not p.is_file() or any(part.startswith('pilot') for part in p.relative_to(out).parts):continue
                if p.name.startswith('background_jacobian') or '.partial.' in p.name:continue
                z.write(p,arcname=Path(out.name)/p.relative_to(out))
        print(f'Archive: {archive.resolve()} ({archive.stat().st_size/1e6:.1f} MB)')
    print(f'Validated {len(completed)}/{expected} reconstructions; report: {(out/"REPORT.md").resolve()}')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=Path('results/column_104mm_256e/run01'))
    ap.add_argument('--allow-partial',action='store_true')
    ap.add_argument('--render-vtk',action='store_true')
    ap.add_argument('--package',action='store_true')
    args=ap.parse_args();finalize(args.results,args.allow_partial,args.render_vtk,args.package)
