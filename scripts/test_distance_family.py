"""Test extension integrity and separation from historical evaluation outputs."""
import importlib.util,json,pathlib,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('distance_family_test',ROOT/'scripts/distance/family.py')
family=importlib.util.module_from_spec(spec);spec.loader.exec_module(family)
class Tests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.temporary.name)
        self.previous=(family.ROOT,family.DATA);family.ROOT=self.root;family.DATA=self.root/'data';family.DATA.mkdir()
        payload=self.root/'checkpoint.pt';payload.write_bytes(b'verified-checkpoint')
        self.extension={'version':'distance-2048-50m-v1','model':'50m','parameters':51927104,'evaluation_models':['5m','10m','25m','50m'],'inputs_sha256':{'checkpoint.pt':family.sha(payload)}}
        self.write()
    def write(self):(family.DATA/'extension-50m.json').write_text(json.dumps(self.extension))
    def tearDown(self):family.ROOT,family.DATA=self.previous;self.temporary.cleanup()
    def test_checkpoint_mutation_rejected(self):
        family.verify('50m');(self.root/'checkpoint.pt').write_bytes(b'changed')
        with self.assertRaises(AssertionError):family.verify('50m')
    def test_wrong_model_size_rejected(self):
        self.extension['parameters']=97352024;self.write()
        with self.assertRaises(AssertionError):family.verify('50m')
    def test_historical_result_directories_preserved(self):
        paths={size:family.report_dir(size) for size in family.SIZES}
        self.assertEqual(paths['5m'],paths['10m']);self.assertEqual(len(set(paths.values())),4)
        self.assertEqual(paths['25m'].name,'distance-2048-25m-v1')
        self.assertEqual(paths['100m'].name,'distance-2048-100m-v1')
if __name__=='__main__':unittest.main()
