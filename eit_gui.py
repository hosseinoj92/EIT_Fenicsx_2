"""Native desktop SI configurator; batch FEniCSx runs stay in a child process."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from eitcore.physical import default_config, normalize_config, material_admittivity


class EITApp:
    def __init__(self, root, dimension=3):
        self.root = root
        self.cfg = normalize_config(default_config(dimension))
        self.vars = {}
        self.events = queue.Queue()
        self.process = None
        self.busy = False
        self.stop_requested = False
        self.wsl_runtime = False
        self.pid_path = None
        root.title("EIT laboratory — SI simulation in 2D and 3D")
        root.geometry("1080x860")
        root.minsize(960, 740)
        style = ttk.Style(root)
        style.configure("TLabel", padding=3)
        bar = ttk.Frame(root, padding=10)
        bar.pack(fill="x")
        ttk.Label(bar, text="EIT laboratory", font=("Segoe UI", 18, "bold")).pack(side="left")
        ttk.Label(bar, text="All lengths in metres • RMS currents and voltages").pack(side="right")
        self.book = ttk.Notebook(root)
        self.book.pack(fill="both", expand=True, padx=12)
        self.geo = ttk.Frame(self.book, padding=12)
        self.physics = ttk.Frame(self.book, padding=12)
        self.rec = ttk.Frame(self.book, padding=12)
        self.advanced = ttk.Frame(self.book, padding=12)
        self.output = ttk.Frame(self.book, padding=12)
        for frame, name in [(self.geo, "Geometry"), (self.physics, "Excitation and materials"),
                            (self.rec, "Reconstruction"), (self.advanced, "Full configuration"), (self.output, "Results")]:
            self.book.add(frame, text=name)
        self.field(self.geo, "Dimension", "dimension", ["2", "3"])
        self.vars["dimension"].trace_add("write", self.dimension_changed)
        for label, key in [("Case name", "name"), ("Cylinder radius [m]", "radius"),
                           ("2D electrodes / 3D electrodes per ring", "count"),
                           ("2D extrusion thickness [m]", "thickness"),
                           ("2D electrode arc width [degree]", "electrode_width_deg"),
                           ("Cylinder height [m] (3D)", "height"), ("Number of rings (3D)", "n_rings"),
                           ("Ring separation [m] (when heights list is blank)", "ring_spacing"),
                           ("Ring heights [m], comma separated; blank centres stack", "ring_heights"),
                           ("Circular electrode diameter [m] (3D)", "electrode_diameter"),
                           ("First-ring rotation [degree]", "electrode_offset_deg"),
                           ("Rotation per ring [degree] (3D)", "ring_rotation_deg"),
                           ("Truth centre element size [m]", "h_center"),
                           ("Truth boundary element size [m]", "h_edge")]:
            self.field(self.geo, label, key)
        ttk.Label(self.geo, text="3D: circular footprints projected onto the curved sidewall.\n"
                  "Rings are centred vertically. Custom heights (ring_z) are available in Full configuration.\n"
                  "Electrodes are numbered counterclockwise, ring by ring from bottom to top.").grid(column=0, row=20, columnspan=2, sticky="w", pady=8)
        ttk.Button(self.geo, text="Preview geometry and material response", command=self.preview).grid(column=0, row=21, columnspan=2, sticky="w")
        self.field(self.physics, "Excitation frequencies [Hz], comma separated", "frequencies")
        self.field(self.physics, "Injected current per pair [A RMS]", "current_a")
        self.field(self.physics, "Drive protocol", "protocol", ["mixed", "adjacent", "opposite", "axial"])
        for label, key in [("Specific contact resistance [ohm m²]", "resistance_ohm_m2"),
                           ("Specific contact capacitance [F/m²]", "capacitance_f_m2"),
                           ("Background conductivity [S/m]", "bg_sigma"),
                           ("Background relative permittivity [1]", "bg_eps"),
                           ("Inclusion conductivity [S/m]", "inc_sigma"),
                           ("Inclusion relative permittivity [1]", "inc_eps"),
                           ("Circular / spherical inclusion radius [m]", "inc_radius"),
                           ("Inclusion centre [m], x,y or x,y,z", "inc_center")]:
            self.field(self.physics, label, key)
        ttk.Label(self.physics, text="The first inclusion is editable here. Add more inclusions, ellipsoids, boxes,\n"
                  "Debye relaxation or individual electrode contacts in Full configuration.\n\n"
                  "Material values are illustrative. Use measured values for your own phantom.\n"
                  "Frequency effects come from permittivity, dielectric relaxation and contact capacitance.\n"
                  "At zero permittivity and zero contact capacitance, resistive voltages do not depend on frequency.\n"
                  "Phasor convention: exp(+jωt). Contacts use a parallel RC model.").grid(column=0, row=20, columnspan=2, sticky="w", pady=10)
        self.field(self.rec, "Run reconstruction", "enabled", ["True", "False"])
        self.field(self.rec, "Reconstruct permittivity as well as conductivity", "reconstruct_permittivity", ["True", "False"])
        self.field(self.rec, "3D algorithms, comma separated; blank uses single solver", "algorithms")
        self.field(self.rec, "Comparison frequencies [Hz], comma separated", "rec_frequencies")
        for label, key in [("Reconstruction frequency [Hz] (must be in sweep)", "frequency_hz"),
                           ("Reconstruction centre element size [m]", "rec_h_center"),
                           ("Reconstruction boundary element size [m]", "rec_h_edge"),
                           ("Maximum iterations", "iterations"),
                           ("Quadratic / GN regularization [dimensionless]", "regularization"),
                           ("Minimum conductivity [S/m]", "sigma_min_s_m"),
                           ("Maximum conductivity [S/m]", "sigma_max_s_m"),
                           ("I/Q noise standard deviation per channel [V RMS]", "noise_std_v")]:
            self.field(self.rec, label, key)
        ttk.Label(self.rec, text="Joint complex Gauss–Newton reconstruction uses a separate mesh and a spatial prior.\n"
                  "At DC only conductivity is observable. With permittivity reconstruction disabled,\n"
                  "permittivity is fixed to the background material. Smaller elements require more memory.\n"
                  "The original 2D algorithm benchmark remains available in simulate.py.").grid(column=0, row=20, columnspan=2, sticky="w", pady=12)
        ttk.Label(self.advanced, text="JSON and form share one configuration. JSON edits are applied on Validate, Save or Run; Apply updates the form immediately.").pack(anchor="w")
        self.json_text = tk.Text(self.advanced, wrap="none", font=("Consolas", 10))
        self.json_text.pack(fill="both", expand=True)
        controls = ttk.Frame(self.advanced)
        controls.pack(fill="x", pady=5)
        ttk.Button(controls, text="Refresh JSON from form", command=self.refresh_json).pack(side="left")
        ttk.Button(controls, text="Apply JSON to form", command=self.apply_json).pack(side="left", padx=8)
        self.log_text = tk.Text(self.output, wrap="word", font=("Consolas", 10), state="disabled")
        self.log_text.pack(fill="both", expand=True)
        self.result_dir = None
        buttons = ttk.Frame(self.output)
        buttons.pack(fill="x", pady=6)
        ttk.Button(buttons, text="Open result folder", command=self.open_results).pack(side="left")
        ttk.Button(buttons, text="Frequency response", command=lambda: self.show_image("frequency_response.png")).pack(side="left", padx=6)
        ttk.Button(buttons, text="Truth slices", command=lambda: self.show_image("truth.png")).pack(side="left")
        ttk.Button(buttons, text="Reconstruction slices", command=lambda: self.show_image("reconstruction.png")).pack(side="left", padx=6)
        ttk.Button(buttons, text="3D reconstruction", command=lambda:self.show_image('reconstruction_3d.png')).pack(side='left')
        choices=ttk.Frame(self.output);choices.pack(fill='x',pady=4)
        self.result_frequency=tk.StringVar();self.result_algorithm=tk.StringVar()
        ttk.Label(choices,text='Saved frequency:').pack(side='left')
        self.result_frequency_box=ttk.Combobox(choices,textvariable=self.result_frequency,state='readonly',width=15)
        self.result_frequency_box.pack(side='left')
        self.result_frequency_box.bind('<<ComboboxSelected>>',lambda event:self.refresh_result_choices(keep_frequency=True))
        ttk.Label(choices,text='Algorithm:').pack(side='left',padx=(12,0))
        self.result_algorithm_box=ttk.Combobox(choices,textvariable=self.result_algorithm,state='readonly',width=15)
        self.result_algorithm_box.pack(side='left')
        footer = ttk.Frame(root, padding=12)
        footer.pack(fill="x")
        for text, cmd in [("Load configuration", self.load), ("Save configuration", self.save),
                          ("Validate", self.validate), ("Run simulation", self.start), ("Stop", self.stop)]:
            ttk.Button(footer, text=text, command=cmd).pack(side="left", padx=4)
        self.status = tk.StringVar(value="Ready — preview, save or run your SI configuration")
        ttk.Label(root, textvariable=self.status).pack(anchor="w", padx=16, pady=(0, 8))
        self.populate()
        self.vars['ring_heights'].trace_add('write', lambda *_: self.dimension_states())
        self.vars['algorithms'].trace_add('write', lambda *_: self.dimension_states())
        self.book.bind('<<NotebookTabChanged>>', self.tab_changed)
        self.active_tab=self.book.select()
        root.after(100, self.drain)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def field(self, parent, label, key, choices=None):
        row = len(parent.grid_slaves())//2
        self.vars[key] = tk.StringVar()
        ttk.Label(parent, text=label).grid(column=0, row=row, sticky="w", padx=4, pady=3)
        widget = ttk.Combobox(parent, textvariable=self.vars[key], values=choices, state="readonly", width=38) if choices else ttk.Entry(parent, textvariable=self.vars[key], width=41)
        widget.grid(column=1, row=row, sticky="ew", padx=10, pady=3)
        parent.columnconfigure(1, weight=1)
        if not hasattr(self, "widgets"):
            self.widgets = {}
        self.widgets[key] = widget

    def populate(self):
        cfg, m, r = self.cfg, self.cfg["mesh"], self.cfg["reconstruction"]
        values = dict(m, **r)
        values.update(name=cfg["name"], count=m["n_electrodes"] if m["dimension"] == 2 else m["electrodes_per_ring"],
                      thickness=cfg["thickness"], frequencies=", ".join(repr(v) for v in cfg["frequencies_hz"]),
                      **cfg["drive"], **cfg["contact"], bg_sigma=cfg["background"]["conductivity_s_m"],
                      bg_eps=cfg["background"].get("relative_permittivity", 0),
                      rec_h_center=r["h_center"], rec_h_edge=r["h_edge"])
        values.update(ring_heights=', '.join(repr(v) for v in (m.get('ring_z') or [])),
                      algorithms=','.join(r.get('algorithms') or []),
                      rec_frequencies=', '.join(repr(v) for v in (r.get('frequencies_hz') or [])))
        # Truth h keys must not be overwritten by reconstruction values.
        values.update(h_center=m["h_center"], h_edge=m["h_edge"])
        inc = cfg.get("inclusions", [])
        simple = bool(inc and inc[0]["shape"] in ("circle", "sphere"))
        self.simple_inclusion = simple
        self.loading = True
        for key, var in self.vars.items():
            var.set(str(values.get(key, "")))
        if simple:
            for key, value in {"inc_sigma": inc[0]["material"]["conductivity_s_m"],
                               "inc_eps": inc[0]["material"].get("relative_permittivity", 0),
                               "inc_radius": inc[0]["radius_m"],
                               "inc_center": ", ".join(map(str, inc[0]["center_m"]))}.items():
                self.vars[key].set(str(value))
        for key in ("inc_sigma", "inc_eps", "inc_radius", "inc_center"):
            self.widgets[key].configure(state="normal" if simple else "disabled")
        self.loading = False
        self.dimension_states()
        self.write_json(cfg)

    def dimension_states(self):
        is3 = self.vars["dimension"].get() == "3"
        for key in ("height", "n_rings", "ring_spacing", "ring_heights", "electrode_diameter", "ring_rotation_deg"):
            self.widgets[key].configure(state="normal" if is3 else "disabled")
        if is3 and self.vars['ring_heights'].get().strip():
            self.widgets['ring_spacing'].configure(state='disabled')
        if 'algorithms' in self.vars:
            comparison=bool(self.vars['algorithms'].get().strip())
            self.widgets['frequency_hz'].configure(state='disabled' if comparison else 'normal')
            self.widgets['rec_frequencies'].configure(state='normal' if comparison else 'disabled')
        for key in ("thickness", "electrode_width_deg"):
            self.widgets[key].configure(state="disabled" if is3 else "normal")

    def dimension_changed(self, *_):
        if getattr(self, "loading", False):
            return
        dim = int(self.vars["dimension"].get())
        self.cfg = normalize_config(default_config(dim))
        self.populate()
        self.status.set(f"Loaded {dim}D defaults; use Load configuration to restore a saved case")

    def collect(self):
        from copy import deepcopy
        text=self.json_text.get('1.0','end-1c')
        if text != getattr(self,'json_baseline',text):
            self.cfg=normalize_config(json.loads(text))
            self.populate()
        cfg = deepcopy(self.cfg)
        v = {k: var.get().strip() for k, var in self.vars.items()}
        m = cfg["mesh"]
        m["dimension"] = int(v["dimension"])
        for key in ("radius", "height", "ring_spacing", "electrode_diameter", "electrode_width_deg", "electrode_offset_deg", "ring_rotation_deg", "h_center", "h_edge"):
            m[key] = float(v[key])
        m['ring_z']=[float(x) for x in v['ring_heights'].split(',')] if v['ring_heights'] else None
        m["n_rings"] = int(v["n_rings"])
        if m['dimension']==3:
            m['electrodes_per_ring']=int(v['count'])
        m["n_electrodes"] = int(v["count"]) * (m["n_rings"] if m["dimension"] == 3 else 1)
        cfg["name"], cfg["thickness"] = v["name"], float(v["thickness"])
        cfg["frequencies_hz"] = [float(x) for x in v["frequencies"].split(",")]
        cfg["drive"].update(current_a=float(v["current_a"]), protocol=v["protocol"])
        for key in ("resistance_ohm_m2", "capacitance_f_m2"):
            # Preserve lists of per-electrode values loaded through JSON.
            cfg["contact"][key] = json.loads(v[key])
        cfg["background"].update(conductivity_s_m=float(v["bg_sigma"]), relative_permittivity=float(v["bg_eps"]))
        if self.simple_inclusion:
            inc = cfg["inclusions"][0]
            inc["radius_m"] = float(v["inc_radius"])
            inc["center_m"] = [float(x) for x in v["inc_center"].split(",")]
            inc["material"].update(conductivity_s_m=float(v["inc_sigma"]), relative_permittivity=float(v["inc_eps"]))
        r = cfg["reconstruction"]
        for key in ("enabled", "reconstruct_permittivity"):
            r[key] = v[key] == "True"
        for key in ("frequency_hz", "regularization", "sigma_min_s_m", "sigma_max_s_m", "noise_std_v"):
            r[key] = float(v[key])
        r.update(iterations=int(v["iterations"]), h_center=float(v["rec_h_center"]), h_edge=float(v["rec_h_edge"]))
        if v['algorithms']:
            r['algorithms']=[x.strip() for x in v['algorithms'].split(',')]
            r['frequencies_hz']=[float(x) for x in v['rec_frequencies'].split(',')] if v['rec_frequencies'] else list(cfg['frequencies_hz'])
            if not r.get('parameter_grid'):
                import numpy as np
                r['parameter_grid']={'nx':12,'ny':12,'z_edges_m':np.linspace(0,m['height'],25).tolist()}
        elif 'algorithms' in r:
            r['algorithms']=None
        return normalize_config(cfg)

    def guarded(self, action):
        try:
            return action()
        except Exception as exc:
            messagebox.showerror("Check configuration", str(exc), parent=self.root)
            return None

    def write_json(self, cfg):
        self.json_text.delete("1.0", "end")
        self.json_text.insert("1.0", json.dumps(cfg, indent=2))
        self.json_baseline=self.json_text.get('1.0','end-1c')

    def tab_changed(self, event=None):
        current=self.book.select()
        if getattr(self,'active_tab',None)==str(self.advanced) and current!=str(self.advanced):
            text=self.json_text.get('1.0','end-1c')
            if text!=self.json_baseline:
                cfg=self.guarded(lambda:normalize_config(json.loads(text)))
                if cfg is None:
                    self.book.select(self.advanced)
                    return
                self.cfg=cfg
                self.populate()
        self.active_tab=current
        if current==str(self.advanced):
            cfg=self.guarded(self.collect)
            if cfg:
                self.write_json(cfg)

    def refresh_json(self):
        cfg = self.guarded(self.collect)
        if cfg:
            self.write_json(cfg)

    def apply_json(self):
        cfg = self.guarded(lambda: normalize_config(json.loads(self.json_text.get("1.0", "end"))))
        if cfg:
            self.cfg = cfg
            self.populate()
            self.status.set("JSON applied")

    def validate(self):
        cfg = self.guarded(self.collect)
        if cfg:
            self.status.set(f"Valid {cfg['mesh']['dimension']}D configuration; {cfg['mesh']['n_electrodes']} electrodes")

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("JSON configuration", "*.json")])
        if path:
            cfg = self.guarded(lambda: normalize_config(json.loads(Path(path).read_text(encoding="utf-8"))))
            if cfg:
                self.cfg = cfg
                self.populate()

    def save(self):
        cfg = self.guarded(self.collect)
        if cfg:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON configuration", "*.json")])
            if path:
                Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                self.status.set(f"Saved {path}")

    def preview(self):
        cfg = self.guarded(self.collect)
        if not cfg:
            return
        import numpy as np
        import matplotlib.pyplot as plt
        from mesh_tools import electrode_centres
        m = cfg["mesh"]
        fig = plt.figure(figsize=(11, 5))
        ax = fig.add_subplot(121, projection="3d" if m["dimension"] == 3 else None)
        centres = electrode_centres(m)
        theta = np.linspace(0, 2*np.pi, 100)
        R = m["radius"]
        if m["dimension"] == 3:
            for z in (0, m["height"]):
                ax.plot(R*np.cos(theta), R*np.sin(theta), z, color="gray")
            for i, (x, y, z) in enumerate(centres):
                # The radial-cylinder footprint is circular in tangent projection.
                s = m["electrode_diameter"]/2*np.cos(theta)
                a = np.arctan2(y, x)+np.arcsin(s/R)
                ax.plot(R*np.cos(a), R*np.sin(a), z+m["electrode_diameter"]/2*np.sin(theta), color="tab:orange")
                ax.text(x, y, z, str(i+1), fontsize=8)
            ax.set(zlabel="z [m]", zlim=(0, m["height"]))
            ax.set_box_aspect((2*R, 2*R, m["height"]))
        else:
            ax.plot(R*np.cos(theta), R*np.sin(theta), color="gray")
            for i, (x, y) in enumerate(centres):
                angle = np.arctan2(y, x)
                a = angle+np.deg2rad(m["electrode_width_deg"])*np.linspace(-.5, .5, 20)
                ax.plot(R*np.cos(a), R*np.sin(a), linewidth=5, color="tab:orange")
                ax.text(x*1.08, y*1.08, str(i+1))
            ax.set_aspect("equal")
        ax.set(xlabel="x [m]", ylabel="y [m]", title="Electrode geometry")
        ax2 = fig.add_subplot(122)
        f = np.asarray(cfg["frequencies_hz"])
        f = np.sort(f)
        gamma = np.array([material_admittivity(cfg["background"], x) for x in f])
        ax2.plot(f, abs(gamma), "o-", label="Background")
        if cfg["inclusions"]:
            g = np.array([material_admittivity(cfg["inclusions"][0]["material"], x) for x in f])
            ax2.plot(f, abs(g), "o-", label="First inclusion")
        if np.all(f > 0):
            ax2.set_xscale("log")
        ax2.set(xlabel="Frequency [Hz]", ylabel="Admittivity magnitude [S/m]", title="Material preview (before FEM solve)")
        ax2.grid(alpha=.3)
        ax2.legend()
        fig.tight_layout()
        plt.show(block=False)

    def start(self):
        if self.busy:
            return
        cfg = self.guarded(self.collect)
        if not cfg:
            return
        self.busy = True
        self.stop_requested = False
        self.result_dir = None
        self.book.select(self.output)
        self.status.set("Starting numerical solver ...")
        root = Path(__file__).resolve().parent
        run_dir = root/"results"/"physical_configs"
        run_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        path = run_dir/f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        threading.Thread(target=self.worker, args=(root, path), daemon=True).start()

    def worker(self, root, path):
        try:
            probe = subprocess.run([sys.executable, "-c", "import dolfinx"], capture_output=True,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if self.stop_requested:
                raise InterruptedError("Simulation cancelled before startup")
            self.pid_path = path.with_suffix(".pid")
            self.wsl_runtime = False
            command = [sys.executable, "-u", str(root/"physical_eit.py"), "--config", str(path),
                       "--worker-pid-file", str(self.pid_path)]
            if probe.returncode and sys.platform == "win32":
                relative = path.relative_to(root).as_posix()
                self.wsl_runtime = True
                command = ["wsl", "--exec", "python3", "-u", "physical_eit.py", "--config", relative,
                           "--worker-pid-file", self.pid_path.relative_to(root).as_posix()]
                self.events.put(("log", "Using FEniCSx in the default WSL distribution.\n"))
            elif probe.returncode:
                raise RuntimeError("FEniCSx is missing. See the README installation instructions.")
            self.events.put(("log", f"Saved configuration: {path}\n"))
            env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
            self.process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8", errors="replace", env=env,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in self.process.stdout:
                self.events.put(("log", line))
                if self.stop_requested:
                    self.terminate_worker()
            code = self.process.wait()
            self.events.put(("done", code))
        except Exception as exc:
            self.events.put(("log", f"Error: {exc}\n"))
            self.events.put(("done", 1))
        finally:
            self.process = None

    def drain(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "done":
                    self.busy = False
                    self.status.set("Simulation complete — use the result buttons" if value == 0 else "Simulation stopped or failed — see log")
                else:
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", value)
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
                    if value.startswith("Results: "):
                        result = value.removeprefix("Results: ").strip()
                        if sys.platform == "win32" and result.startswith("/mnt/"):
                            result = result[5].upper()+":"+result[6:].replace("/", "\\")
                        self.result_dir = Path(result)
                        self.refresh_result_choices()
        except queue.Empty:
            pass
        self.root.after(100, self.drain)

    def show_image(self, name):
        path=self.result_dir/name if self.result_dir else None
        if path is not None and not path.exists():
            if name=='truth.png' and (self.result_dir/'truth_sections.png').exists():
                path=self.result_dir/'truth_sections.png'
            else:
                mapped='reconstruction_sections.png' if name in ('truth.png','reconstruction.png') else name
                path=self.result_dir/'reconstructions'/self.result_frequency.get()/self.result_algorithm.get()/mapped
        if path is None or not path.exists():
            self.status.set("Run a simulation first; reconstruction images require reconstruction enabled")
            return
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.imshow(plt.imread(path))
        ax.axis("off")
        fig.tight_layout()
        plt.show(block=False)

    def refresh_result_choices(self,keep_frequency=False):
        if not self.result_dir:
            return
        parent=self.result_dir/'reconstructions'
        frequencies=sorted([p.name for p in parent.glob('*Hz') if p.is_dir()],
                           key=lambda name:float(name[:-2]))
        self.result_frequency_box.configure(values=frequencies)
        if not keep_frequency or self.result_frequency.get() not in frequencies:
            preferred=f"{self.cfg['reconstruction']['frequency_hz']:g}Hz"
            self.result_frequency.set(preferred if preferred in frequencies else frequencies[0] if frequencies else '')
        algorithms=[p.name for p in (parent/self.result_frequency.get()).glob('*')
                    if p.is_dir() and (p/'reconstruction.vtu').exists()]
        self.result_algorithm_box.configure(values=algorithms)
        if self.result_algorithm.get() not in algorithms:
            self.result_algorithm.set('tv' if 'tv' in algorithms else algorithms[0] if algorithms else '')

    def open_results(self):
        if self.result_dir:
            if sys.platform == "win32":
                os.startfile(self.result_dir)
            else:
                subprocess.Popen(["xdg-open", str(self.result_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        self.stop_requested = True
        self.terminate_worker()

    def terminate_worker(self):
        process = self.process
        if process and process.poll() is None:
            if self.wsl_runtime:
                if self.pid_path and self.pid_path.exists():
                    pid = int(self.pid_path.read_text(encoding="ascii"))
                    subprocess.run(["wsl", "--exec", "kill", "-TERM", str(pid)],
                                   capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                process.terminate()

    def close(self):
        self.stop()
        if self.busy:
            self.status.set("Stopping solver before closing ...")
            self.root.after(100, self.close)
        else:
            self.root.destroy()


def launch(dimension=3):
    import matplotlib
    matplotlib.use("TkAgg")
    root = tk.Tk()
    EITApp(root, dimension)
    root.mainloop()


if __name__ == "__main__":
    launch()
