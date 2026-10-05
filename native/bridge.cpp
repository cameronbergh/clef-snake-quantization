// SPDX-License-Identifier: Apache-2.0
#include "llama.h"
#include "ggml-backend.h"
#include <cstring>
#include <string>
#include <exception>

struct clef_bridge { llama_model * model; llama_context * ctx; int width; int batch; };
static thread_local std::string last_error;
extern "C" const char * clef_error() { return last_error.c_str(); }
extern "C" void * clef_load(const char * path, int capacity, int gpu_layers) {
    try {
        ggml_backend_load_all();
        llama_backend_init();
        auto mp=llama_model_default_params();
        mp.n_gpu_layers=gpu_layers;
        auto * model=llama_model_load_from_file(path,mp);
        if(!model) {last_error="model loading failed";return nullptr;}
        auto cp=llama_context_default_params();
        cp.n_ctx=4096; cp.n_batch=capacity; cp.n_ubatch=capacity; cp.n_seq_max=1;
        cp.n_threads=6; cp.n_threads_batch=6;
        cp.embeddings=true; cp.pooling_type=LLAMA_POOLING_TYPE_NONE;
        cp.attention_type=LLAMA_ATTENTION_TYPE_CAUSAL;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv=true;
        auto * ctx=llama_init_from_model(model,cp);
        if(!ctx) {llama_model_free(model);last_error="context creation failed";return nullptr;}
        return new clef_bridge{model,ctx,llama_model_n_embd_out(model),capacity};
    }catch(const std::exception & e){last_error=e.what();return nullptr;}
}
extern "C" int clef_width(void * handle) {return static_cast<clef_bridge*>(handle)->width;}
extern "C" int clef_hidden(void * handle,const int32_t * tokens,int n,float * output) {
    auto * b=static_cast<clef_bridge*>(handle);
    if(n<1 || n>b->batch){last_error="token length outside batch capacity";return -1;}
    llama_memory_clear(llama_get_memory(b->ctx),true);
    auto batch=llama_batch_init(n,0,1);batch.n_tokens=n;
    for(int i=0;i<n;i++){
        batch.token[i]=tokens[i];batch.pos[i]=i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;
        batch.logits[i]=1; // output final hidden state for every input token
    }
    const int rc=llama_decode(b->ctx,batch);
    if(rc){llama_batch_free(batch);last_error="llama_decode error "+std::to_string(rc);return rc;}
    for(int i=0;i<n;i++){
        const auto * raw=llama_get_embeddings_ith(b->ctx,i);
        if(!raw){llama_batch_free(batch);last_error="missing token hidden state";return -2;}
        std::memcpy(output+i*b->width,raw,b->width*sizeof(float));
    }
    llama_batch_free(batch);return 0;
}
extern "C" void clef_free(void * handle) {
    if(!handle)return;
    auto * b=static_cast<clef_bridge*>(handle);llama_free(b->ctx);llama_model_free(b->model);delete b;
}
