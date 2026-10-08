// PhysReal-X tests for the constrained forward and Newton/PCG projection.
#include "physrealx_articulated_untangling.cuh"
#include <iostream>

int main() {
    try {
        constexpr unsigned count = 12;
        nlohmann::json description = {{"physrealx_articulation", {
            {"canonical_vertices", 4}, {"moving_begin", 2},
            {"pinned_canonical_vertices", {1}},
            {"rotations", {1,0,0,0,1,0,0,0,1, 0,-1,0,1,0,0,0,0,1,
                           1,0,0,0,0,-1,0,1,0}}
        }}};
        physrealx_unfolding::configure(description, count);
        auto apply = [&](const std::vector<double>& x) {
            double* d = nullptr;
            if (cudaMalloc(&d, x.size()*sizeof(double)) != cudaSuccess)
                throw std::runtime_error("CUDA allocation failed");
            cudaMemcpy(d,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice);
            physrealx_unfolding::project(d,count);
            std::vector<double> y(x.size());
            auto error = cudaMemcpy(y.data(),d,y.size()*sizeof(double),cudaMemcpyDeviceToHost);
            cudaFree(d);
            if (error != cudaSuccess) throw std::runtime_error("CUDA projection failed");
            return y;
        };
        std::vector<double> x(3*count), y(3*count);
        for (unsigned i=0;i<x.size();++i) {x[i]=std::sin(i);y[i]=std::cos(i*2.3);}
        auto px=apply(x), py=apply(y), ppx=apply(px);
        double left=0,right=0;
        for(unsigned i=0;i<x.size();++i) {
            if(std::abs(px[i]-ppx[i])>1e-12) throw std::runtime_error("projection not idempotent");
            left+=x[i]*py[i]; right+=px[i]*y[i];
        }
        if(std::abs(left-right)>1e-12) throw std::runtime_error("projection not self adjoint");
        for(unsigned p=0;p<3;++p) for(unsigned d=0;d<3;++d)
            if(px[d*count+p*4+1]!=0) throw std::runtime_error("protected vertex drift");
        // The projected gradient is a descent direction in the feasible subspace.
        double descent=0,norm=0;
        for(unsigned i=0;i<x.size();++i) {descent+=x[i]*px[i];norm+=px[i]*px[i];}
        if(std::abs(descent-norm)>1e-12 || descent<=0) throw std::runtime_error("invalid descent direction");
        double *gx=nullptr, *dx=nullptr;
        cudaMalloc(&gx, x.size()*sizeof(double));
        cudaMalloc(&dx, x.size()*sizeof(double));
        cudaMemcpy(gx,px.data(),px.size()*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(dx,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice);
        if(!physrealx_unfolding::positive_direction(gx,dx,count))
            throw std::runtime_error("valid inexact Newton direction rejected");
        for(auto& v:x) v=-v;
        cudaMemcpy(dx,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice);
        if(physrealx_unfolding::positive_direction(gx,dx,count))
            throw std::runtime_error("ascending Newton direction accepted");
        x[0]=std::numeric_limits<double>::quiet_NaN();
        cudaMemcpy(dx,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice);
        if(physrealx_unfolding::positive_direction(gx,dx,count))
            throw std::runtime_error("nonfinite Newton direction accepted");
        cudaFree(gx); cudaFree(dx);
        bool rejected=false;
        description["physrealx_articulation"]["rotations"][0]=2;
        try {physrealx_unfolding::configure(description,count);} catch(const std::runtime_error&) {rejected=true;}
        if(!rejected) throw std::runtime_error("invalid rotation accepted");
        std::cout << "canonical CUDA projection: idempotence, adjoint, pins, descent, validation PASSED\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
