#pragma once

#include <opencv2/opencv.hpp>

namespace MI
{

double Cost(const std::vector<uint8_t> & grayValues,
            const std::vector<uint8_t> & reflectanceValues,
            int bins = 256,
            bool draw = false,
            std::string debugImagesPrefixPath = "");

} // namespace MI
