// Stage a complete group of activations in shared panels. Unlike the 32x8
// tile this sweeps each packed weight exactly once, even for M=64.
template<int Codec,int Block,int Bytes,int Rows,int M>
__global__ void dense_gemv_panel_visit(inference_cuda_dense_gemv_args a){
    __shared__ float x[M][64];
    int t=threadIdx.x,lane=t%32,warp=t/32,n0=blockIdx.x*(8*Rows)+warp*Rows;
    auto *weights=static_cast<const uint8_t *>(a.w);
    float sum[Rows][M]={};
    for(int k0=0;k0<a.cols;k0+=64){
#pragma unroll
        for(int i=t;i<M*64;i+=256){int m=i/64,k=i%64;x[m][k]=m<a.m?static_cast<const float *>(a.x)[size_t(m)*a.ld_x+k0+k]:0.f;}
        __syncthreads();
#pragma unroll
        for(int j=0;j<2;j++){
            int k=k0+32*j+lane;
#pragma unroll
            for(int n=0;n<Rows;n++){
                auto *b=weights+((n0+n)*(a.cols/Block)+k/Block)*Bytes;
                float w=fifi_gguf::decode<Codec>(b,k%Block);
#pragma unroll
                for(int m=0;m<M;m++) sum[n][m]=fmaf(w,x[m][j*32+lane],sum[n][m]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int n=0;n<Rows;n++){
#pragma unroll
        for(int m=0;m<M;m++){
            float v=warp_sum(sum[n][m]);if(!lane && m<a.m) static_cast<float *>(a.y)[size_t(m)*a.ld_y+n0+n]=v;
        }
    }
}
