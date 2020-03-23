#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <map>

#include "FeatureDetector.h"

using namespace pw;

const char program_describe[] =
        "helper program to organization and use 3d reconstruction pipelines\n"
        "\n"
        "Usage: pipeline_3d <command> [<args>]\n"
        "Available options:\n"
        "  -h [ --help ]         display this help and exit\n"
        "Available command:";

int FeatureMatch(const int argc BOOST_ATTRIBUTE_UNUSED, const char *argv[] BOOST_ATTRIBUTE_UNUSED){
  return EXIT_SUCCESS;
}

std::map<std::string, int (*)(const int, const char *[])> commandMap = {
  {"FeatureDetect", &FeatureDetector::run},
  {"FeatureMatch", &FeatureMatch}
};

int main(const int argc, const char *argv[])
{
  if((argc <= 1) or !commandMap.count(argv[1])){
    std::cout << program_describe << std::endl;
    std::cout << "  ";
    for(const auto &it: commandMap){
      std::cout << it.first << " ";
    }
    std::cout << std::endl;
    if((argc > 1) and (!strcmp(argv[1], "-h") or !strcmp(argv[1], "--help"))){
      return EXIT_SUCCESS;
    } else {
      return EXIT_FAILURE;
    }
  }

  return commandMap[argv[1]](argc - 1, argv + 1);
}