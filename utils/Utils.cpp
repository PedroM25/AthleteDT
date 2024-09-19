#include "Utils.h"

namespace utils{

bool readLinesIntoVector(const std::string filePath, std::vector<std::string>& vectorToFill){
    std::ifstream file(filePath);
    
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            vectorToFill.push_back(line);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file \'" << filePath << "\'" << std::endl;
        return false;
    }
    return true;
}

}
