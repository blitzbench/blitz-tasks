#include <iostream>

#include "cpp_task_demo.hpp"

int main(int argc, const char **argv) {
    cpp_task_demo::CppTaskDemo task;
    std::cout << "Blitz-Task C++ Demo Application\n"
              << "====================================" << std::endl;
    blitz::Callbacks cb{
        .on_status=[](blitz::Status status) { std::cout << "Received status " << status << std::endl; },
        .on_start=[]{ std::cout << "Started..." << std::endl; },
        .on_progress=[](const blitz::Metric& metric) { std::cout << "\r -> " << metric.value << " " << metric.unit; },
        .on_complete=[](const std::vector<blitz::Metric>& metrics) {
            std::cout << "\n => Done: ";
            for (const auto& metric : metrics) {
                std::cout << "  " << metric.direction << " | " << metric.name << " | " << metric.value << " " << metric.unit << "\n";
            }
            std::cout << std::flush;
        },
        .on_error=[](blitz::Result result, const std::string& msg) { std::cout << "Received error " << result << ": " << msg << std::endl; }
    };
    task.run(cb);
}
