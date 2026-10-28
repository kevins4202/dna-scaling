import pathlib
import runpy

ROOT=pathlib.Path(__file__).resolve().parents[1]

if __name__=='__main__':
    target=ROOT/'artifacts/cnn-v1/score.py'
    if not target.is_file():
        raise SystemExit('The final project CNN has not been released. Pilot scoring is historical and is not used automatically.')
    runpy.run_path(str(target),run_name='__main__')
