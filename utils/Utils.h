#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

namespace utils{

/*
* Read from file supported objects that model can identify
*/
bool readLinesIntoVector(const std::string filePath, std::vector<std::string>& vectorToFill);

}
