# SPDX-License-Identifier: Apache-2.0
"""Raw all-token llama.cpp backbone states -> unchanged official JointSchemaHead."""
import ctypes, json, os, pathlib, sys, time
import numpy as np
import torch
import gguf
from safetensors.torch import load_file
from transformers import AutoProcessor

ROOT=pathlib.Path(__file__).resolve().parent
OFFICIAL=pathlib.Path(os.environ['CLEF_OFFICIAL_DIR']).expanduser().resolve()
sys.path.insert(0,str(OFFICIAL))
from joint_schema_model import JointSchemaHead, systemone, collate_records, encode_record

class RawBackbone:
    def __init__(self,path,capacity=2048):
        self.path=pathlib.Path(path)
        self.lib=ctypes.CDLL(str(pathlib.Path(os.environ['CLEF_BRIDGE_LIBRARY']).expanduser().resolve()))
        lib=self.lib
        lib.clef_load.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int];lib.clef_load.restype=ctypes.c_void_p
        lib.clef_width.argtypes=[ctypes.c_void_p];lib.clef_width.restype=ctypes.c_int
        lib.clef_hidden.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int32),ctypes.c_int,ctypes.POINTER(ctypes.c_float)];lib.clef_hidden.restype=ctypes.c_int
        lib.clef_error.restype=ctypes.c_char_p
        lib.clef_free.argtypes=[ctypes.c_void_p]
        self.handle=lib.clef_load(str(self.path).encode(),capacity,99)
        if not self.handle:raise RuntimeError(lib.clef_error().decode())
        self.width=lib.clef_width(self.handle);self.capacity=capacity
        assert self.width==4096,self.width
    def hidden(self,input_ids):
        tokens=np.asarray(input_ids,dtype=np.int32)
        assert tokens.ndim==1 and len(tokens)<=self.capacity
        output=np.empty((len(tokens),self.width),dtype=np.float32)
        rc=self.lib.clef_hidden(self.handle,tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),len(tokens),output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc:raise RuntimeError(self.lib.clef_error().decode())
        if not np.isfinite(output).all():raise RuntimeError('Non-finite raw backbone states')
        return output
    def close(self):
        if self.handle:self.lib.clef_free(self.handle);self.handle=None

class OutputRows:
    """Exact dequantized GGUF output rows, consumed only via indexing by official head."""
    def __init__(self,path,device='mps',dtype=torch.bfloat16):
        self.reader=gguf.GGUFReader(path)
        self.tensor=next(t for t in self.reader.tensors if t.name=='output.weight')
        assert list(self.tensor.shape)==[4096,248320],self.tensor.shape
        self.device=device;self.dtype=dtype;self.cache={}
    def numpy_rows(self,ids):
        data=self.tensor.data[np.asarray(ids,dtype=np.int64)]
        return gguf.quants.dequantize(data,self.tensor.tensor_type)
    def __getitem__(self,ids):
        values=ids.detach().cpu().tolist() if isinstance(ids,torch.Tensor) else list(ids)
        missing=sorted(set(values)-self.cache.keys())
        if missing:
            rows=torch.from_numpy(self.numpy_rows(missing).copy()).to(device=self.device,dtype=self.dtype)
            for token,row in zip(missing,rows):self.cache[token]=row
        return torch.stack([self.cache[token] for token in values])

class QuantClef(torch.nn.Module):
    def __init__(self,path):
        super().__init__()
        config=json.loads((OFFICIAL/'joint_head_config.json').read_text())
        self.head=JointSchemaHead(**config)
        self.head.load_state_dict(load_file(OFFICIAL/'joint_head.safetensors'),strict=True)
        self.head=self.head.to(device='mps',dtype=torch.bfloat16).eval()
        self.backbone=RawBackbone(path)
        self.output_rows=OutputRows(path)
        self.timings={}
        self.eval()
    def forward(self,batch):
        assert len(batch['records'])==1 and not batch.get('media')
        encoded=batch['records'][0]
        start=time.perf_counter();raw=self.backbone.hidden(encoded.input_ids)
        backbone_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter()
        hidden=torch.from_numpy(raw).unsqueeze(0).to(device='mps',dtype=torch.bfloat16)
        logits=self.head(hidden,batch['input_ids'],batch['attention_mask'],batch['records'],self.output_rows)
        torch.mps.synchronize()
        self.timings={'backbone_ms':round(backbone_ms,3),'transfer_and_official_head_ms':round((time.perf_counter()-start)*1000,3)}
        return logits

def load(path):
    processor=AutoProcessor.from_pretrained(OFFICIAL,local_files_only=True)
    return QuantClef(path),processor
