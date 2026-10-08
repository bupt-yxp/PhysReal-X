#pragma once

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace mu::config {

struct ParsedDescription {
    nlohmann::json json;
    std::vector<std::filesystem::path> meshPaths;
    nlohmann::json energy;
    std::optional<std::filesystem::path> energyLogFile;
    std::string solverType;
};

inline std::optional<ParsedDescription> LoadDescriptionFile(
    const std::filesystem::path& descriptionPath) {
    if (descriptionPath.empty()) {
        spdlog::error("Configuration path is empty.");
        return std::nullopt;
    }

    std::ifstream inputDescription(descriptionPath);
    if (!inputDescription.is_open()) {
        spdlog::error("Failed to open {}", descriptionPath.string());
        return std::nullopt;
    }

    nlohmann::json descriptionJson;
    try {
        descriptionJson = nlohmann::json::parse(inputDescription);
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse {}: {}", descriptionPath.string(), e.what());
        return std::nullopt;
    }

    const bool hasSolverPasses = descriptionJson.contains("solverPasses") &&
                                 descriptionJson["solverPasses"].is_array() &&
                                 !descriptionJson["solverPasses"].empty();

    if (!descriptionJson.contains("filePath") || !descriptionJson["filePath"].is_array()) {
        spdlog::error("Description file is missing entry 'filePath' or it is not an array.");
        return std::nullopt;
    }

    nlohmann::json energyValue;
    if (descriptionJson.contains("energy")) {
        const auto& energyJson = descriptionJson["energy"];
        if (!energyJson.is_string() && !energyJson.is_array()) {
            spdlog::error("Description file entry 'energy' must be a string or an array of strings or objects.");
            return std::nullopt;
        }
        energyValue = energyJson;
    } else if (!hasSolverPasses) {
        spdlog::error("Description file is missing entry 'energy'.");
        return std::nullopt;
    } else {
        const auto& passes = descriptionJson["solverPasses"];
        const auto& firstPass = passes.front();
        if (firstPass.contains("energy")) {
            energyValue = firstPass["energy"];
        } else {
            energyValue = nlohmann::json{};
        }
    }

    std::string solverTypeValue;
    if (descriptionJson.contains("solverType") && descriptionJson["solverType"].is_string()) {
        solverTypeValue = descriptionJson["solverType"].get<std::string>();
    } else if (!hasSolverPasses) {
        spdlog::error("Description file is missing entry 'solverType' or it is not a string.");
        return std::nullopt;
    } else {
        const auto& passes = descriptionJson["solverPasses"];
        if (!passes.empty()) {
            const auto& firstPass = passes.front();
            if (firstPass.contains("solverType") && firstPass["solverType"].is_string()) {
                solverTypeValue = firstPass["solverType"].get<std::string>();
            } else {
                solverTypeValue = "GradientDescent";
            }
        } else {
            solverTypeValue = "GradientDescent";
        }
    }

    const auto& filePathArray = descriptionJson["filePath"];
    if (filePathArray.empty()) {
        spdlog::error("Description file entry 'filePath' is empty.");
        return std::nullopt;
    }

    ParsedDescription parsed;
    parsed.json = descriptionJson;
    parsed.energy = energyValue;
    parsed.solverType = std::move(solverTypeValue);

    parsed.meshPaths.reserve(filePathArray.size());
    for (const auto& filePathValue : filePathArray) {
        if (!filePathValue.is_string()) {
            spdlog::error("Each 'filePath' entry must be a string.");
            return std::nullopt;
        }
        parsed.meshPaths.emplace_back(filePathValue.get<std::string>());
    }

    if (descriptionJson.contains("energyLogFile")) {
        const auto& logValue = descriptionJson["energyLogFile"];
        if (!logValue.is_string()) {
            spdlog::error("'energyLogFile' must be a string if provided.");
            return std::nullopt;
        }
        const auto logPathString = logValue.get<std::string>();
        if (!logPathString.empty()) {
            parsed.energyLogFile = std::filesystem::path(logPathString);
        }
    }

    return parsed;
}

}
