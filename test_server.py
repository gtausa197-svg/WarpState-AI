"""Run with pytest. CPU preparation tests plus a small CUDA full-state resume test."""
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import types
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('server_pipeline', ROOT/'run_40b.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def args(tmp_path):
    return types.SimpleNamespace(work_dir=tmp_path, tokenizer=ROOT/'tokenizer.model',
        dataset='fixture',dataset_config='fixture',revision='main',train_tokens=300,val_tokens=100,
        workers=2,documents_per_batch=7,test_small_gpu=True,seq_len=8)


def test_masked_final_batch_and_wrap():
    data = np.arange(19,dtype=np.uint16)
    x,y = pipeline.batches(data,17,2,4,5,torch.device('cpu'))
    assert x.flatten().tolist() == [17,18,0,1,2,3,4,5]
    assert y.flatten().tolist() == [18,0,1,2,3,-100,-100,-100]


def test_preparation_resume_truncates_uncommitted_tail(tmp_path,monkeypatch):
    a = args(tmp_path)
    holdout = next('document '+str(i) for i in range(100000) if pipeline.is_holdout('document '+str(i)))
    normal = next('other '+str(i) for i in range(100) if not pipeline.is_holdout('other '+str(i)))
    documents = [{'text':holdout if i%3==0 else normal} for i in range(500)]
    class Source:
        def __init__(self): self.offset=0
        def __iter__(self):
            while self.offset < len(documents):
                row=documents[self.offset]
                self.offset+=1
                yield row
        def state_dict(self): return {'offset':self.offset}
        def load_state_dict(self,state): self.offset=state['offset']
    monkeypatch.setitem(sys.modules,'datasets',types.SimpleNamespace(__version__='fixture',load_dataset=lambda *a,**kw:Source()))
    monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(HfApi=lambda:types.SimpleNamespace(dataset_info=lambda *a,**kw:types.SimpleNamespace(sha='fixed-commit'))))
    pipeline.STOP=True
    assert not pipeline.prepare(a)
    with (tmp_path/'data/prepare-state.pkl').open('rb') as f:
        state=pickle.load(f)
    with (tmp_path/'data/train.bin').open('ab') as f: f.write(b'uncommitted-tail')
    pipeline.STOP=False
    assert pipeline.prepare(a)
    pipeline.check_manifest(a)
    other=tmp_path/'continuous'
    b=args(other)
    assert pipeline.prepare(b)
    for name in ('train.bin','val.bin'):
        assert (tmp_path/'data'/name).read_bytes() == (other/'data'/name).read_bytes()
    assert state['iterator']['offset'] == 7
    a.train_tokens+=1
    with pytest.raises(ValueError): pipeline.check_manifest(a)


def test_memory_cap_refuses_insufficient_headroom(monkeypatch):
    a = types.SimpleNamespace(vram_limit_gib=33.5, vram_headroom_gib=3, autocast_cache=False)
    calls = []
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (36*1024**3, 96*1024**3))
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match='requested allocator cap and headroom'):
        pipeline.configure_memory(a)
    assert not calls
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (37*1024**3, 96*1024**3))
    pipeline.configure_memory(a)
    assert calls == [(33.5/96, torch.device('cuda:0'))]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('initialization', ['weights', 'fresh'])
def test_gpu_full_state_resume_and_exact_partial_update(tmp_path, initialization):
    sys.path.insert(0,str(ROOT))
    from warpstate import WarpStateConfig,WarpStateLM
    cfg=WarpStateConfig(vocab_size=32768,d_model=32,n_heads=2,n_cores=2,logical_depth=4,
                        ffn_hidden=64,chunk_size=8,parallel_memory=True)
    config_path=tmp_path/'config.json'
    cfg.to_json(config_path)
    torch.manual_seed(133)
    original=WarpStateLM(cfg)
    weights=tmp_path/'initial.pt'
    torch.save({'config':vars(cfg),'model':original.state_dict(),'tokens_seen':100},weights)
    dirs=[tmp_path/'continuous',tmp_path/'resumed']
    common=[sys.executable,str(ROOT/'run_40b.py'),'--no-bootstrap','--test-small-gpu',
            '--stage','train','--config',str(config_path),
            '--train-tokens','73','--val-tokens','256','--seq-len','8','--micro-batch','2',
            '--grad-accum','2','--loss-chunk-size','5','--workers','2','--eval-batches','2','--eval-every','1',
            '--no-autocast-cache','--vram-limit-gib','0.5','--vram-headroom-gib','1']
    common += ['--weights',str(weights)] if initialization == 'weights' else ['--from-scratch']
    for directory in dirs:
        (directory/'data').mkdir(parents=True)
        np.arange(73,dtype='<u2').tofile(directory/'data/train.bin')
        np.arange(256,dtype='<u2').tofile(directory/'data/val.bin')
        a=args(directory)
        a.dataset='HuggingFaceFW/fineweb-edu'; a.dataset_config='sample-100BT'
        a.train_tokens=73; a.val_tokens=256
        (directory/'data/manifest.json').write_text(json.dumps({'complete':True,'signature':pipeline.data_signature(a)}))
    def run(directory,extra=[]):
        result=subprocess.run(common+['--work-dir',str(directory)]+extra,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    run(dirs[0])
    run(dirs[1],['--max-steps','1'])
    run(dirs[1])
    a=torch.load(dirs[0]/'run/latest.pt',weights_only=False)
    b=torch.load(dirs[1]/'run/latest.pt',weights_only=False)
    assert a['tokens_seen']==b['tokens_seen']==73
    assert a['step']==b['step']==3
    assert a['origin'] == b['origin']
    if initialization == 'fresh':
        assert a['origin'] == {'initialization': 'random', 'seed': 1337, 'previous_training_tokens': 0}
    for key in a['model']: torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    for index,state in a['optimizer']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(value,b['optimizer']['state'][index][key],rtol=0,atol=0)
    changed=subprocess.run(common+['--work-dir',str(dirs[1]),'--lr','0.0002'],capture_output=True,text=True)
    assert changed.returncode!=0 and 'Resume settings/data differ' in changed.stderr
