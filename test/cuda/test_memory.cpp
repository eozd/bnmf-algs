#include "../catch2.hpp"
#include "../ctor_dtor_counter.hpp"
#include "cuda/memory.hpp"
#include "defs.hpp"
#include <numeric>
#include <vector>

using C = Counter<int>;

using namespace bnmf_algs;

TEST_CASE("Test HostMemory1D", "[cuda_memory]") {
    constexpr size_t N = 50;

    SECTION("HostMemory1D makes no allocation/copying/destruction") {
        std::vector<C> vec(N);

        {
            C::reset_counters();
            cuda::HostMemory1D<C> host_mem(vec.data(), N);

            REQUIRE(C::TotalDefaultCtorCount == 0);

            auto host_mem_copy = host_mem;
            host_mem_copy = host_mem_copy;

            REQUIRE(C::TotalCopyCount == 0);

            auto host_mem_move(std::move(host_mem));
            host_mem_move = std::move(host_mem_move);

            REQUIRE(C::TotalMoveCount == 0);
        }

        REQUIRE(C::TotalDtorCount == 0);
    }
}

TEST_CASE("Test DeviceMemory1D", "[cuda_memory]") {
    constexpr size_t N = 50;

    SECTION("Move constructor") {
        std::vector<int> src(N);
        std::vector<int> src_copy(N);
        cuda::HostMemory1D<int> src_mem(src.data(), N);
        cuda::HostMemory1D<int> src_copy_mem(src_copy.data(), N);

        // fill source array
        std::iota(src.begin(), src.end(), 0);

        // copy source to GPU
        cuda::DeviceMemory1D<int> old_gpu_mem(N);
        cuda::copy1D(old_gpu_mem, src_mem);

        // move GPU memory to a new object (memory is not moved)
        auto new_gpu_mem = std::move(old_gpu_mem);

        // old DeviceMemory1D has lost ownership
        REQUIRE(old_gpu_mem.data() == nullptr);

        // copy back from new DeviceMemory1D
        cuda::copy1D(src_copy_mem, new_gpu_mem);

        REQUIRE(std::equal(src.begin(), src.end(), src_copy.begin()));
    }

    SECTION("Move assignment operator") {
        std::vector<int> src(N);
        std::vector<int> src_copy(N);
        cuda::HostMemory1D<int> src_mem(src.data(), N);
        cuda::HostMemory1D<int> src_copy_mem(src_copy.data(), N);

        // fill source array
        std::iota(src.begin(), src.end(), 0);

        // copy source to GPU
        cuda::DeviceMemory1D<int> old_gpu_mem(N);
        cuda::copy1D(old_gpu_mem, src_mem);

        // allocate new GPU memory
        cuda::DeviceMemory1D<int> new_gpu_mem(N);

        // move old memory onto new one
        new_gpu_mem = std::move(old_gpu_mem);

        // old DeviceMemory1D has lost ownership
        REQUIRE(old_gpu_mem.data() == nullptr);

        // copy back from new DeviceMemory1D
        cuda::copy1D(src_copy_mem, new_gpu_mem);

        REQUIRE(std::equal(src.begin(), src.end(), src_copy.begin()));
    }
}

TEST_CASE("Test HostMemory2D", "[cuda_memory]") {
    constexpr size_t N = 50;

    SECTION("HostMemory2D makes no allocation/copying/destruction") {
        matrix_t<C> mat(N, N);

        {
            C::reset_counters();
            cuda::HostMemory2D<C> host_mem(mat.data(), N, N);

            REQUIRE(C::TotalDefaultCtorCount == 0);

            auto host_mem_copy = host_mem;
            host_mem_copy = host_mem_copy;

            REQUIRE(C::TotalCopyCount == 0);

            auto host_mem_move(std::move(host_mem));
            host_mem_move = std::move(host_mem_move);

            REQUIRE(C::TotalMoveCount == 0);
        }

        REQUIRE(C::TotalDtorCount == 0);
    }
}

TEST_CASE("Test DeviceMemory2D", "[cuda_memory]") {
    constexpr size_t N = 50;

    SECTION("Move constructor") {
        const matrixd src = matrixd::Random(N, N);
        matrixd src_copy(N, N);
        cuda::HostMemory2D<const double> src_mem(src.data(), N, N);
        cuda::HostMemory2D<double> src_copy_mem(src_copy.data(), N, N);

        // copy source to GPU
        cuda::DeviceMemory2D<double> old_gpu_mem(N, N);
        cuda::copy2D(old_gpu_mem, src_mem);

        // move GPU memory to a new object (memory is not moved)
        auto new_gpu_mem = std::move(old_gpu_mem);

        // old DeviceMemory2D has lost ownership
        REQUIRE(old_gpu_mem.data() == nullptr);

        // copy back from new DeviceMemory2D
        cuda::copy2D(src_copy_mem, new_gpu_mem);

        REQUIRE(src.isApprox(src_copy));
    }

    SECTION("Move assignment operator") {
        const matrixd src = matrixd::Random(N, N);
        matrixd src_copy(N, N);
        cuda::HostMemory2D<const double> src_mem(src.data(), N, N);
        cuda::HostMemory2D<double> src_copy_mem(src_copy.data(), N, N);

        // copy source to GPU
        cuda::DeviceMemory2D<double> old_gpu_mem(N, N);
        cuda::copy2D(old_gpu_mem, src_mem);

        // allocate new GPU memory
        cuda::DeviceMemory2D<double> new_gpu_mem(N, N);

        // move old gpu memory to new memory
        new_gpu_mem = std::move(old_gpu_mem);

        // old DeviceMemory2D has lost ownership
        REQUIRE(old_gpu_mem.data() == nullptr);

        // copy back from new DeviceMemory2D
        cuda::copy2D(src_copy_mem, new_gpu_mem);

        REQUIRE(src.isApprox(src_copy));
    }
}

TEST_CASE("Test HostMemory3D", "[cuda_memory]") {
    constexpr size_t N = 50;

    SECTION("HostMemory3D makes no allocation/copying/destruction") {
        tensor_t<C, 3> tensor(N, N, N);

        {
            C::reset_counters();
            cuda::HostMemory3D<C> host_mem(tensor.data(), N, N, N);

            REQUIRE(C::TotalDefaultCtorCount == 0);

            auto host_mem_copy = host_mem;
            host_mem_copy = host_mem_copy;

            REQUIRE(C::TotalCopyCount == 0);

            auto host_mem_move(std::move(host_mem));
            host_mem_move = std::move(host_mem_move);

            REQUIRE(C::TotalMoveCount == 0);
        }

        REQUIRE(C::TotalDtorCount == 0);
    }
}

TEST_CASE("Test DeviceMemory3D", "[cuda_memory]") {
    constexpr size_t N = 50;

    SECTION("Move constructor") {
        tensor_t<int, 3> src(N, N, N);
        tensor_t<int, 3> src_copy(N, N, N);
        cuda::HostMemory3D<int> src_mem(src.data(), N, N, N);
        cuda::HostMemory3D<int> src_copy_mem(src_copy.data(), N, N, N);

        // fill source array
        std::iota(src.data(), src.data() + N * N * N, 0);

        // copy source to GPU
        cuda::DeviceMemory3D<int> old_gpu_mem(N, N, N);
        cuda::copy3D(old_gpu_mem, src_mem);

        // move GPU memory to a new object (memory is not moved)
        auto new_gpu_mem = std::move(old_gpu_mem);

        // old DeviceMemory3D has lost ownership
        REQUIRE(old_gpu_mem.pitched_ptr().ptr == nullptr);

        // copy back from new DeviceMemory3D
        cuda::copy3D(src_copy_mem, new_gpu_mem);

        REQUIRE(
            std::equal(src.data(), src.data() + N * N * N, src_copy.data()));
    }

    SECTION("Move assignment operator") {
        tensor_t<int, 3> src(N, N, N);
        tensor_t<int, 3> src_copy(N, N, N);
        cuda::HostMemory3D<int> src_mem(src.data(), N, N, N);
        cuda::HostMemory3D<int> src_copy_mem(src_copy.data(), N, N, N);

        // fill source array
        std::iota(src.data(), src.data() + N * N * N, 0);

        // copy source to GPU
        cuda::DeviceMemory3D<int> old_gpu_mem(N, N, N);
        cuda::copy3D(old_gpu_mem, src_mem);

        // allocate a new GPU memory
        cuda::DeviceMemory3D<int> new_gpu_mem(N, N, N);

        // move old GPU memory to new one
        new_gpu_mem = std::move(old_gpu_mem);

        // old DeviceMemory3D has lost ownership
        REQUIRE(old_gpu_mem.pitched_ptr().ptr == nullptr);

        // copy back from new DeviceMemory3D
        cuda::copy3D(src_copy_mem, new_gpu_mem);

        REQUIRE(
            std::equal(src.data(), src.data() + N * N * N, src_copy.data()));
    }
}

TEST_CASE("Test infer_copy_kind 1D", "[cuda_memory]") {
    using namespace cuda;
    SECTION("From host to host") {
        constexpr auto a =
            infer_copy_kind<HostMemory1D<double>, HostMemory1D<int>,
                            HostMemory1D, DeviceMemory1D>();

        REQUIRE(a == cudaMemcpyHostToHost);

        constexpr auto b =
            infer_copy_kind<HostMemory1D<float>, HostMemory1D<double>,
                            HostMemory1D, DeviceMemory1D>();

        REQUIRE(b == cudaMemcpyHostToHost);
    }

    SECTION("From host to device") {
        constexpr auto a =
            infer_copy_kind<DeviceMemory1D<double>, HostMemory1D<int>,
                            HostMemory1D, DeviceMemory1D>();

        REQUIRE(a == cudaMemcpyHostToDevice);

        constexpr auto b =
            infer_copy_kind<DeviceMemory1D<float>, HostMemory1D<double>,
                            HostMemory1D, DeviceMemory1D>();

        REQUIRE(b == cudaMemcpyHostToDevice);
    }

    SECTION("From device to host") {
        constexpr auto a =
            infer_copy_kind<HostMemory1D<double>, DeviceMemory1D<int>,
                            HostMemory1D, DeviceMemory1D>();

        REQUIRE(a == cudaMemcpyDeviceToHost);

        constexpr auto b =
            infer_copy_kind<HostMemory1D<float>, DeviceMemory1D<double>,
                            HostMemory1D, DeviceMemory1D>();

        REQUIRE(b == cudaMemcpyDeviceToHost);
    }

    SECTION("From device to device") {
        constexpr auto a =
            infer_copy_kind<DeviceMemory1D<double>, DeviceMemory1D<int>,
                            HostMemory1D, DeviceMemory1D>();

        REQUIRE(a == cudaMemcpyDeviceToDevice);

        constexpr auto b =
            infer_copy_kind<DeviceMemory1D<float>, DeviceMemory1D<double>,
                            HostMemory1D, DeviceMemory1D>();

        REQUIRE(b == cudaMemcpyDeviceToDevice);
    }
}

TEST_CASE("Test infer_copy_kind 2D", "[cuda_memory]") {
    using namespace cuda;
    SECTION("From host to host") {
        constexpr auto a =
            infer_copy_kind<HostMemory2D<double>, HostMemory2D<int>,
                            HostMemory2D, DeviceMemory2D>();

        REQUIRE(a == cudaMemcpyHostToHost);

        constexpr auto b =
            infer_copy_kind<HostMemory2D<float>, HostMemory2D<double>,
                            HostMemory2D, DeviceMemory2D>();

        REQUIRE(b == cudaMemcpyHostToHost);
    }

    SECTION("From host to device") {
        constexpr auto a =
            infer_copy_kind<DeviceMemory2D<double>, HostMemory2D<int>,
                            HostMemory2D, DeviceMemory2D>();

        REQUIRE(a == cudaMemcpyHostToDevice);

        constexpr auto b =
            infer_copy_kind<DeviceMemory2D<float>, HostMemory2D<double>,
                            HostMemory2D, DeviceMemory2D>();

        REQUIRE(b == cudaMemcpyHostToDevice);
    }

    SECTION("From device to host") {
        constexpr auto a =
            infer_copy_kind<HostMemory2D<double>, DeviceMemory2D<int>,
                            HostMemory2D, DeviceMemory2D>();

        REQUIRE(a == cudaMemcpyDeviceToHost);

        constexpr auto b =
            infer_copy_kind<HostMemory2D<float>, DeviceMemory2D<double>,
                            HostMemory2D, DeviceMemory2D>();

        REQUIRE(b == cudaMemcpyDeviceToHost);
    }

    SECTION("From device to device") {
        constexpr auto a =
            infer_copy_kind<DeviceMemory2D<double>, DeviceMemory2D<int>,
                            HostMemory2D, DeviceMemory2D>();

        REQUIRE(a == cudaMemcpyDeviceToDevice);

        constexpr auto b =
            infer_copy_kind<DeviceMemory2D<float>, DeviceMemory2D<double>,
                            HostMemory2D, DeviceMemory2D>();

        REQUIRE(b == cudaMemcpyDeviceToDevice);
    }
}

TEST_CASE("Test infer_copy_kind 3D", "[cuda_memory]") {
    using namespace cuda;
    SECTION("From host to host") {
        constexpr auto a =
            infer_copy_kind<HostMemory3D<double>, HostMemory3D<int>,
                            HostMemory3D, DeviceMemory3D>();

        REQUIRE(a == cudaMemcpyHostToHost);

        constexpr auto b =
            infer_copy_kind<HostMemory3D<float>, HostMemory3D<double>,
                            HostMemory3D, DeviceMemory3D>();

        REQUIRE(b == cudaMemcpyHostToHost);
    }

    SECTION("From host to device") {
        constexpr auto a =
            infer_copy_kind<DeviceMemory3D<double>, HostMemory3D<int>,
                            HostMemory3D, DeviceMemory3D>();

        REQUIRE(a == cudaMemcpyHostToDevice);

        constexpr auto b =
            infer_copy_kind<DeviceMemory3D<float>, HostMemory3D<double>,
                            HostMemory3D, DeviceMemory3D>();

        REQUIRE(b == cudaMemcpyHostToDevice);
    }

    SECTION("From device to host") {
        constexpr auto a =
            infer_copy_kind<HostMemory3D<double>, DeviceMemory3D<int>,
                            HostMemory3D, DeviceMemory3D>();

        REQUIRE(a == cudaMemcpyDeviceToHost);

        constexpr auto b =
            infer_copy_kind<HostMemory3D<float>, DeviceMemory3D<double>,
                            HostMemory3D, DeviceMemory3D>();

        REQUIRE(b == cudaMemcpyDeviceToHost);
    }

    SECTION("From device to device") {
        constexpr auto a =
            infer_copy_kind<DeviceMemory3D<double>, DeviceMemory3D<int>,
                            HostMemory3D, DeviceMemory3D>();

        REQUIRE(a == cudaMemcpyDeviceToDevice);

        constexpr auto b =
            infer_copy_kind<DeviceMemory3D<float>, DeviceMemory3D<double>,
                            HostMemory3D, DeviceMemory3D>();

        REQUIRE(b == cudaMemcpyDeviceToDevice);
    }
}

TEST_CASE("Test copy1D", "[cuda_memory]") {
    constexpr size_t N = 50;

    SECTION("From host to host") {
        std::vector<int> src(N);
        std::vector<int> dst(N);

        cuda::HostMemory1D<int> src_mem(src.data(), N);
        cuda::HostMemory1D<int> dst_mem(dst.data(), N);

        // fill source array
        std::iota(src.begin(), src.end(), 0);

        // copy from host to host
        cuda::copy1D(dst_mem, src_mem);

        REQUIRE(std::equal(src.begin(), src.end(), dst.begin()));
    }

    SECTION("Between host and device") {
        std::vector<int> src(N);
        std::vector<int> src_copy(N);

        // fill source array
        std::iota(src.begin(), src.end(), 0);

        cuda::HostMemory1D<int> src_mem(src.data(), N);
        cuda::HostMemory1D<int> src_copy_mem(src_copy.data(), N);
        cuda::DeviceMemory1D<int> gpu_mem(N);

        // copy from host to device
        cuda::copy1D(gpu_mem, src_mem);

        // copy from device to back to host
        cuda::copy1D(src_copy_mem, gpu_mem);

        REQUIRE(std::equal(src.begin(), src.end(), src_copy.begin()));
    }

    SECTION("From device to device") {
        std::vector<int> src(N);
        std::vector<int> src_copy(N);

        // fill source array
        std::iota(src.begin(), src.end(), 0);

        cuda::HostMemory1D<int> src_mem(src.data(), N);
        cuda::HostMemory1D<int> src_copy_mem(src_copy.data(), N);
        cuda::DeviceMemory1D<int> first_dst_mem(N);
        cuda::DeviceMemory1D<int> second_dst_mem(N);

        // copy from host to first device array
        cuda::copy1D(first_dst_mem, src_mem);

        // copy from device to device
        cuda::copy1D(second_dst_mem, first_dst_mem);

        // copy from device to back to host
        cuda::copy1D(src_copy_mem, second_dst_mem);

        REQUIRE(std::equal(src.begin(), src.end(), src_copy.begin()));
    }
}

TEST_CASE("Test copy2D", "[cuda_memory]") {
    constexpr size_t N = 50;

    SECTION("From host to host") {
        const matrixd src = matrixd::Random(N, N);
        matrixd dst(N, N);

        cuda::HostMemory2D<const double> src_mem(src.data(), N, N);
        cuda::HostMemory2D<double> dst_mem(dst.data(), N, N);

        // copy from host to host
        cuda::copy2D(dst_mem, src_mem);

        REQUIRE(src.isApprox(dst));
    }

    SECTION("Between host and device") {
        const matrixd src = matrixd::Random(N, N);
        matrixd src_copy(N, N);

        cuda::HostMemory2D<const double> src_mem(src.data(), N, N);
        cuda::HostMemory2D<double> src_copy_mem(src_copy.data(), N, N);
        cuda::DeviceMemory2D<double> gpu_mem(N, N);

        // copy from host to device
        cuda::copy2D(gpu_mem, src_mem);

        // copy from device to back to host
        cuda::copy2D(src_copy_mem, gpu_mem);

        REQUIRE(src.isApprox(src_copy));
    }

    SECTION("From device to device") {
        const matrixd src = matrixd::Random(N, N);
        matrixd src_copy(N, N);

        cuda::HostMemory2D<const double> src_mem(src.data(), N, N);
        cuda::HostMemory2D<double> src_copy_mem(src_copy.data(), N, N);
        cuda::DeviceMemory2D<double> first_dst_mem(N, N);
        cuda::DeviceMemory2D<double> second_dst_mem(N, N);

        // copy from host to first device array
        cuda::copy2D(first_dst_mem, src_mem);

        // copy from device to device
        cuda::copy2D(second_dst_mem, first_dst_mem);

        // copy from device to back to host
        cuda::copy2D(src_copy_mem, second_dst_mem);

        REQUIRE(src.isApprox(src_copy));
    }
}

TEST_CASE("Test copy3D", "[cuda_memory]") {
    constexpr size_t N = 50;

    SECTION("From host to host") {
        tensor_t<int, 3> src(N, N, N);
        tensor_t<int, 3> dst(N, N, N);

        cuda::HostMemory3D<int> src_mem(src.data(), N, N, N);
        cuda::HostMemory3D<int> dst_mem(dst.data(), N, N, N);

        // fill source array
        std::iota(src.data(), src.data() + N * N * N, 0);

        // copy from host to host
        cuda::copy3D(dst_mem, src_mem);

        REQUIRE(std::equal(src.data(), src.data() + N * N * N, dst.data()));
    }

    SECTION("Between host and device") {
        tensor_t<int, 3> src(N, N, N);
        tensor_t<int, 3> src_copy(N, N, N);

        // fill source array
        std::iota(src.data(), src.data() + N * N * N, 0);

        cuda::HostMemory3D<int> src_mem(src.data(), N, N, N);
        cuda::HostMemory3D<int> src_copy_mem(src_copy.data(), N, N, N);
        cuda::DeviceMemory3D<int> gpu_mem(N, N, N);

        // copy from host to device
        cuda::copy3D(gpu_mem, src_mem);

        // copy from device to back to host
        cuda::copy3D(src_copy_mem, gpu_mem);

        REQUIRE(
            std::equal(src.data(), src.data() + N * N * N, src_copy.data()));
    }

    SECTION("From device to device") {
        tensor_t<int, 3> src(N, N, N);
        tensor_t<int, 3> src_copy(N, N, N);

        // fill source array
        std::iota(src.data(), src.data() + N * N * N, 0);

        cuda::HostMemory3D<int> src_mem(src.data(), N, N, N);
        cuda::HostMemory3D<int> src_copy_mem(src_copy.data(), N, N, N);
        cuda::DeviceMemory3D<int> first_dst_mem(N, N, N);
        cuda::DeviceMemory3D<int> second_dst_mem(N, N, N);

        // copy from host to first device array
        cuda::copy3D(first_dst_mem, src_mem);

        // copy from device to device
        cuda::copy3D(second_dst_mem, first_dst_mem);

        // copy from device to back to host
        cuda::copy3D(src_copy_mem, second_dst_mem);

        REQUIRE(
            std::equal(src.data(), src.data() + N * N * N, src_copy.data()));
    }
}
