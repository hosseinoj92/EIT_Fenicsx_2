"""Verify that the native form and authoritative JSON cannot silently diverge."""
import json
from pathlib import Path
import sys
import unittest
import tkinter as tk
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eit_gui import EITApp
from eitcore.physical import default_config,normalize_config


class GUIConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root=tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest(str(exc))
        cls.root.withdraw()
        cls.app=EITApp(cls.root,3)

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def test_exact_roundtrip(self):
        cases=[default_config(2),default_config(3),json.loads(Path('examples/column_104mm_256e.json').read_text())]
        for cfg in cases:
            cfg=normalize_config(cfg)
            self.app.cfg=cfg;self.app.populate()
            self.assertEqual(self.app.collect(),cfg)

    def test_unapplied_json_is_authoritative(self):
        cfg=normalize_config(json.loads(Path('examples/column_104mm_256e.json').read_text()))
        cfg['drive']['current_a']=.000734567891234567
        self.app.json_text.delete('1.0','end')
        self.app.json_text.insert('1.0',json.dumps(cfg))
        self.assertEqual(self.app.collect(),cfg)
        self.assertEqual(float(self.app.vars['current_a'].get()),cfg['drive']['current_a'])

    def test_full_precision_and_explicit_heights(self):
        cfg=default_config(3)
        cfg['frequencies_hz']=[1234.567891234567]
        cfg['reconstruction']['frequency_hz']=cfg['frequencies_hz'][0]
        cfg['mesh']['ring_z']=[.03234567891234567,.07123456789123456]
        cfg=normalize_config(cfg)
        self.app.cfg=cfg;self.app.populate()
        self.assertEqual(self.app.collect(),cfg)
        self.assertTrue(self.app.widgets['ring_spacing'].instate(['disabled']))
        self.app.vars['ring_heights'].set('')
        self.assertFalse(self.app.widgets['ring_spacing'].instate(['disabled']))

    def test_leaving_json_updates_visible_form(self):
        self.app.cfg=normalize_config(default_config(3));self.app.populate()
        self.app.book.select(self.app.advanced);self.root.update()
        cfg=self.app.collect();cfg['drive']['current_a']=.00083
        self.app.json_text.delete('1.0','end');self.app.json_text.insert('1.0',json.dumps(cfg))
        self.app.book.select(self.app.physics);self.root.update()
        self.assertEqual(float(self.app.vars['current_a'].get()),.00083)
        self.assertEqual(self.app.collect(),cfg)


if __name__=='__main__':
    unittest.main(verbosity=2)
