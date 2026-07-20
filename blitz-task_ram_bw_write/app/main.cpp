#include <iostream>

#include "ram_bw_write.hpp"

int main(int argc, const char **argv) {
    ram_bw_write::RamBwWrite task;
    std::cout << "Blitz-Task: Memory write bandwidth\n"
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
    task.run(cb);
}
