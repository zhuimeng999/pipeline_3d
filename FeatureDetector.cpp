//
// Created by lucius on 3/22/20.
//

#include "FeatureDetector.h"
#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <omp.h>

#ifndef HAVE_OPENCV_XFEATURES2D
#error "this module need opencv xfeatures2d enabled"
#endif

namespace pw {
const static std::set<std::string> imageTypeList = {".jpg", ".JPG", ".png", ".PNG"};
static auto _t1 = std::chrono::high_resolution_clock::now();
#define TIMER_START() {_t1 = std::chrono::high_resolution_clock::now();}
#define TIMER_END() {auto duration = std::chrono::duration_cast<std::chrono::seconds> \
        ( std::chrono::high_resolution_clock::now() - _t1 ).count(); \
        std::cout << "time used " << duration << "s" << std::endl;}

int FeatureDetector::run(const int argc, const char *argv[]) {
  boost::program_options::options_description od;
  od.add_options()("image_directory", "image directory")
  ("output_filename", "output filename");
  boost::program_options::positional_options_description pod;
  pod.add("image_directory", 1);
  pod.add("output_filename", 1);

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(od).positional(pod)
                                        .run(), vm);
  boost::program_options::notify(vm);


  if ((!vm.count("image_directory")) or (!vm.count("output_filename"))) {
    std::cout << "you must privide a image directory and a output_filename" << std::endl;
    return EXIT_FAILURE;
  }

  boost::filesystem::path p(vm["image_directory"].as<std::string>());
  if (!exists(p)) {
    std::cout << "the image directory " << p.string() << " does not exist" << std::endl;
    return EXIT_FAILURE;
  } else if (!is_directory(p)) {
    std::cout << p.string() << " is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<boost::filesystem::path> images;

  for (auto &&x : boost::filesystem::directory_iterator(p)) {
    if (imageTypeList.count(x.path().extension().string())) {
      images.push_back(x.path());
    }
  }

  std::sort(images.begin(), images.end());

  //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
  int minHessian = 400;
  std::vector<std::vector<cv::KeyPoint>> keypointsList(images.size());
  std::vector<cv::Mat> descriptorsList(images.size());

  std::cout << "feature detect progress";
  boost::progress_display show_progress(images.size());
  TIMER_START();
#pragma omp parallel default(none) shared(minHessian, show_progress, images, keypointsList, descriptorsList)
  {
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(minHessian);
#pragma omp for
    for (int i = 0; i < images.size(); i++) {
      cv::Mat img = cv::imread(images[i].string(), cv::IMREAD_UNCHANGED);
      detector->detectAndCompute(img, cv::noArray(), keypointsList[i], descriptorsList[i]);
#pragma omp critical
      ++show_progress;
    }
  }
  TIMER_END();

  std::vector<std::pair<int, int>> ompKeys;
  ompKeys.reserve(images.size() * (images.size() - 1) / 2);
  for (int i = 0; i < images.size(); i++) {
    for (int j = i + 1; j < images.size(); j++) {
      ompKeys.emplace_back(i, j);
    }
  }
  std::cout << "feature matching progress";
  show_progress.restart(ompKeys.size());
  std::vector<std::vector<cv::DMatch>> allMatches(ompKeys.size());
  TIMER_START();
#pragma omp parallel default(none) shared(ompKeys, show_progress, allMatches, descriptorsList)
  {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
#pragma omp for
    for (int i = 0; i < ompKeys.size(); i++) {
      std::vector<std::vector<cv::DMatch> > knn_matches;
      matcher->knnMatch(descriptorsList[ompKeys[i].first], descriptorsList[ompKeys[i].second], knn_matches, 2);
      //-- Filter matches using the Lowe's ratio test
      const float ratio_thresh = 0.7f;
      auto & good_matches = allMatches[i];
      for (const auto & knn_matche : knn_matches) {
        if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
          good_matches.push_back(knn_matche[0]);
        }
      }
#pragma omp critical
      ++show_progress;
    }
  }
  TIMER_END();

  boost::filesystem::ofstream of(vm["output_filename"].as<std::string>());
  for(int i = 0; i < ompKeys.size(); i++){
    const auto & good_matches = allMatches[i];
    if (good_matches.size() >= 16) {
      /* Write the pair */
      of << ompKeys[i].first << " " << ompKeys[i].second << std::endl;

      /* Write the number of matches */
      of << good_matches.size() << std::endl;

      for (const auto &good_matche : good_matches) {
        of << good_matche.queryIdx << " " << good_matche.trainIdx << std::endl;
      }
    }
  }
  of.flush();
  of.close();
  return EXIT_SUCCESS;
}
}
