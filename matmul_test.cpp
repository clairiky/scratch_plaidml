#include <iostream>
#include <plaidml/op/op.h>
#include <plaidml/edsl/edsl.h>
#include <plaidml/exec/exec.h>
#include <plaidml/core/core.h>
#include <variant>
#include <memory>
#include <cassert>

using MultiBuffer = std::variant< //
    std::vector<float>,           //
    std::vector<double>,          //
    std::vector<int>,             //
    std::vector<std::int32_t>,    //
    std::vector<std::int64_t>,    //
    std::vector<std::uint32_t>,   //
    std::vector<std::uint64_t>>;

template <typename T>
void compareBuffers(plaidml::View view, const std::vector<T> &expected)
{
    assert(view.size() == expected.size() * sizeof(expected[0]));
    auto data = reinterpret_cast<T *>(view.data());
    std::vector<T> actual(data, data + expected.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin()))
        std::cout << "Equal" << std::endl;
    else
        std::cout << "Not equal" << std::endl;
}
plaidml::edsl::Program makeProgram(const std::string &name, const std::vector<plaidml::edsl::Tensor> &outputs)
{
    auto program = plaidml::edsl::ProgramBuilder(name, outputs).compile();
    std::cout << program << std::endl;
    return program;
}

void checkProgram(
    const plaidml::edsl::Program &program,
    const std::map<plaidml::edsl::TensorRef, MultiBuffer> &inputs,
    const std::map<plaidml::edsl::TensorRef, MultiBuffer> &expected)
{
    auto binder = plaidml::exec::Binder(program);
    auto executable = binder.compile();
    for (const auto &kvp : inputs)
    {
        std::visit([&](auto &&vec) { binder.input(kvp.first).copy_from(vec.data()); }, kvp.second);
    }
    executable->run();
    for (auto kvp : expected)
    {
        auto view = binder.output(kvp.first).mmap_current();
        std::visit([&](auto &&vec) { compareBuffers(view, vec); }, kvp.second);
    }
}

plaidml::edsl::Tensor MatMul(const plaidml::edsl::Tensor &A, const plaidml::edsl::Tensor &B)
{
    plaidml::edsl::TensorDim I, J, K;
    plaidml::edsl::TensorIndex i, j, k;

    A.bind_dims(I, K);
    B.bind_dims(K, J);
    auto C = plaidml::edsl::TensorOutput(I, J);
    C(i, j) += A(i, k) * B(k, j);
    return C;
}

int main()
{
    plaidml::init();
    plaidml::edsl::init();
    plaidml::op::init();
    plaidml::exec::init();
    plaidml::Settings::set("PLAIDML_DEVICE", "llvm_cpu.0");
    plaidml::Settings::set("PLAIDML_TARGET", "llvm_cpu");

    auto A = plaidml::edsl::Placeholder(plaidml::DType::FLOAT32, {3, 3});
    auto B = plaidml::edsl::Placeholder(plaidml::DType::FLOAT32, {3, 3});

    auto C = MatMul(A, B);
    auto program = makeProgram("matmul", {C});

    std::vector<float> input = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };

    std::vector<float> expected = {
        30.0, 36.0, 42.0,
        66.0, 81.0, 96.0,
        102.0, 126.0, 150.0,
    };

    checkProgram(program, {{A, input}, {B, input}}, {{C, expected}});

    std::cout << "done" << std::endl;
}
