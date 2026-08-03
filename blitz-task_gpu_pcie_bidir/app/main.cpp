#include <iostream>
#include <vector>

#include "gpu_pcie_bidir.hpp"

#include <gpgpu/runtime.hpp>

int main(int argc, const char **argv) {
    gpu_pcie_bidir::GpuPcieBidir task;
    std::cout << "Blitz-Task: GPU PCIe bidirectional bandwidth\n"
              << "====================================" << std::endl;
    blitz::Callbacks cb{
        .on_status=[](blitz::Status status) { std::cout << "Received status " << status << std::endl; },
        .on_start=[]{ std::cout << "Started..." << std::endl; },
        .on_progress=[](const blitz::Metric& metric) { std::cout << "\r -> " << metric.value << " " << metric.unit << "        " << std::flush; },
        .on_complete=[](const std::vector<blitz::Metric>& metrics) {
            std::cout << "\n => Done: ";
            for (const auto& metric : metrics) {
                std::cout << "  " << metric.direction << " | " << metric.name << " | " << metric.value << " " << metric.unit << "\n";
                for (const auto& [key, value] : metric.info) {
                    std::cout << "        " << key << ": " << value << "\n";
                }
            }
            std::cout << std::flush;
        },
        .on_error=[](blitz::Result result, const std::string& msg) { std::cout << "Received error " << result << ": " << msg << std::endl; }
    };

    const gpgpu::Report report = gpgpu::Runtime::query();
    if (report.setups().empty()) {
        std::cout << "No GPU setup found." << std::endl;
        return 1;
    }
    const std::vector<bench::gpu::ProbeResult> probed = task.probeSetups(report.setups());
    std::cout << "Probe results (best first):" << std::endl;
    for (const auto& p : probed) {
        std::cout << "  " << p.setup.device.name()
                  << " [" << gpgpu::to_string(p.setup.backend.id()) << "]: ";
        if (p.result.correct) std::cout << p.result.score << " GB/s";
        else std::cout << "failed: " << (p.result.error.empty() ? "incorrect result" : p.result.error);
        std::cout << std::endl;
    }

    // Pin each device's best probed setup and run the full benchmark on it.
    for (const gpgpu::Device& device : report.devices()) {
        for (const auto& p : probed) {
            if (p.setup.device.id() != device.id()) continue;
            if (!p.result.correct) {
                std::cout << "\nSkipping " << device.name() << ": no working backend" << std::endl;
                break;
            }
            std::cout << "\nBenchmarking " << device.name()
                      << " [" << gpgpu::to_string(p.setup.backend.id()) << "]" << std::endl;
            if (task.setSetup(p.setup) != BLITZ_OK) {
                std::cout << "Setup no longer available." << std::endl;
                break;
            }
            task.run(cb);
            break;
        }
    }
}
