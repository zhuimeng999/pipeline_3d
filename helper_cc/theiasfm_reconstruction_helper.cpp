/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>
#include <cstring>
#include <cerrno>
#include <unordered_map>
#include <Eigen/Eigen>
#include "theiasfm_reconstruction_helper.h"

#define PREBUNDLE_SIGNATURE "MVE_PREBUNDLE\n"
#define PREBUNDLE_SIGNATURE_LEN 14

#define SURVEY_SIGNATURE "MVE_SURVEY\n"
#define SURVEY_SIGNATURE_LEN 11

/* ------------------ Input/Output for Prebundle ------------------ */

void
save_prebundle_data (ViewportList const& viewports,
                     PairwiseMatching const& matching, std::ostream& out)
{
  /* Write signature. */
  out.write(PREBUNDLE_SIGNATURE, PREBUNDLE_SIGNATURE_LEN);

  /* Write number of viewports. */
  int32_t num_viewports = static_cast<int32_t>(viewports.size());
  out.write(reinterpret_cast<char const*>(&num_viewports), sizeof(int32_t));

  /* Write per-viewport data. */
  for (std::size_t i = 0; i < viewports.size(); ++i)
  {
    FeatureSet const& vpf = viewports[i].features;

    /* Write positions. */
    int32_t num_positions = static_cast<int32_t>(vpf.positions.size());
    out.write(reinterpret_cast<char const*>(&num_positions), sizeof(int32_t));
    for (std::size_t j = 0; j < vpf.positions.size(); ++j)
      out.write(reinterpret_cast<char const*>(&vpf.positions[j]), sizeof(math::Vec2f));

    /* Write colors. */
    int32_t num_colors = static_cast<int32_t>(vpf.colors.size());
    out.write(reinterpret_cast<char const*>(&num_colors), sizeof(int32_t));
    for (std::size_t j = 0; j < vpf.colors.size(); ++j)
      out.write(reinterpret_cast<char const*>(&vpf.colors[j]), sizeof(math::Vec3uc));
  }

  /* Write number of matching pairs. */
  int32_t num_pairs = static_cast<int32_t>(matching.size());
  out.write(reinterpret_cast<char const*>(&num_pairs), sizeof(int32_t));

  /* Write per-matching pair data. */
  for (std::size_t i = 0; i < matching.size(); ++i)
  {
    TwoViewMatching const& tvr = matching[i];
    int32_t id1 = static_cast<int32_t>(tvr.view_1_id);
    int32_t id2 = static_cast<int32_t>(tvr.view_2_id);
    int32_t num_matches = static_cast<int32_t>(tvr.matches.size());
    out.write(reinterpret_cast<char const*>(&id1), sizeof(int32_t));
    out.write(reinterpret_cast<char const*>(&id2), sizeof(int32_t));
    out.write(reinterpret_cast<char const*>(&num_matches), sizeof(int32_t));
    for (std::size_t j = 0; j < tvr.matches.size(); ++j)
    {
      CorrespondenceIndex const& c = tvr.matches[j];
      int32_t i1 = static_cast<int32_t>(c.first);
      int32_t i2 = static_cast<int32_t>(c.second);
      out.write(reinterpret_cast<char const*>(&i1), sizeof(int32_t));
      out.write(reinterpret_cast<char const*>(&i2), sizeof(int32_t));
    }
  }
}

void
load_prebundle_data (std::istream& in, ViewportList* viewports,
                     PairwiseMatching* matching)
{
  /* Read and check file signature. */
  char signature[PREBUNDLE_SIGNATURE_LEN + 1];
  in.read(signature, PREBUNDLE_SIGNATURE_LEN);
  signature[PREBUNDLE_SIGNATURE_LEN] = '\0';
  if (std::string(PREBUNDLE_SIGNATURE) != signature)
    throw std::invalid_argument("Invalid prebundle file signature");

  viewports->clear();
  matching->clear();

  /* Read number of viewports. */
  int32_t num_viewports;
  in.read(reinterpret_cast<char*>(&num_viewports), sizeof(int32_t));
  viewports->resize(num_viewports);

  /* Read per-viewport data. */
  for (int i = 0; i < num_viewports; ++i)
  {
    FeatureSet& vpf = viewports->at(i).features;

    /* Read positions. */
    int32_t num_positions;
    in.read(reinterpret_cast<char*>(&num_positions), sizeof(int32_t));
    vpf.positions.resize(num_positions);
    for (int j = 0; j < num_positions; ++j)
      in.read(reinterpret_cast<char*>(&vpf.positions[j]), sizeof(math::Vec2f));

    /* Read colors. */
    int32_t num_colors;
    in.read(reinterpret_cast<char*>(&num_colors), sizeof(int32_t));
    vpf.colors.resize(num_colors);
    for (int j = 0; j < num_colors; ++j)
      in.read(reinterpret_cast<char*>(&vpf.colors[j]), sizeof(math::Vec3uc));
  }

  /* Read number of matching pairs. */
  int32_t num_pairs;
  in.read(reinterpret_cast<char*>(&num_pairs), sizeof(int32_t));

  /* Read per-matching pair data. */
  for (int32_t i = 0; i < num_pairs; ++i)
  {
    int32_t id1, id2, num_matches;
    in.read(reinterpret_cast<char*>(&id1), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&id2), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&num_matches), sizeof(int32_t));

    TwoViewMatching tvr;
    tvr.view_1_id = static_cast<int>(id1);
    tvr.view_2_id = static_cast<int>(id2);
    tvr.matches.reserve(num_matches);
    for (int32_t j = 0; j < num_matches; ++j)
    {
      int32_t i1, i2;
      in.read(reinterpret_cast<char*>(&i1), sizeof(int32_t));
      in.read(reinterpret_cast<char*>(&i2), sizeof(int32_t));
      CorrespondenceIndex c;
      c.first = static_cast<int>(i1);
      c.second = static_cast<int>(i2);
      tvr.matches.push_back(c);
    }
    matching->push_back(tvr);
  }
}

void
save_prebundle_to_file (ViewportList const& viewports,
                        PairwiseMatching const& matching, std::string const& filename)
{
  std::ofstream out(filename.c_str(), std::ios::binary);
  if (!out.good())
    throw std::invalid_argument("can not open file " + filename);
  save_prebundle_data(viewports, matching, out);
  out.close();
}

void
load_prebundle_from_file (std::string const& filename,
                          ViewportList* viewports, PairwiseMatching* matching)
{
  std::ifstream in(filename.c_str(), std::ios::binary);
  if (!in.good())
    throw std::invalid_argument("can not open file " + filename);

  try
  {
    load_prebundle_data(in, viewports, matching);
  }
  catch (...)
  {
    in.close();
    throw;
  }

  if (in.eof())
  {
    in.close();
    throw std::runtime_error("Premature EOF");
  }
  in.close();
}

namespace
{
enum BundleFormat
{
  BUNDLE_FORMAT_PHOTOSYNTHER,
  BUNDLE_FORMAT_NOAHBUNDLER,
  BUNDLE_FORMAT_ERROR
};
}  // namespace

void
load_bundler_ps_intern (std::string const& filename, BundleFormat format)
{
  std::ifstream in(filename.c_str());
  if (!in.good())
    throw std::invalid_argument("can not open file " + filename);

  /* Read version information in the first line. */
  std::string version_string;
  std::getline(in, version_string);

  std::string parser_string;
  if (format == BUNDLE_FORMAT_PHOTOSYNTHER)
  {
    parser_string = "Photosynther";
    if (version_string != "drews 1.0")
      format = BUNDLE_FORMAT_ERROR;
  }
  else if (format == BUNDLE_FORMAT_NOAHBUNDLER)
  {
    parser_string = "Bundler";
    if (version_string != "# Bundle file v0.3")
      format = BUNDLE_FORMAT_ERROR;
  }
  else
    throw std::runtime_error("Invalid parser format");

  if (format == BUNDLE_FORMAT_ERROR)
    throw std::runtime_error("Invalid file signature: " + version_string);

  /* Read number of cameras and number of points. */
  int num_views = 0;
  int num_features = 0;
  in >> num_views >> num_features;

  if (in.eof())
    throw std::runtime_error("Unexpected EOF in bundle file");

  if (num_views < 0 || num_views > 10000
      || num_features < 0 || num_features > 100000000)
    throw std::runtime_error("Spurious amount of cameras or features");

  /* Print message according to detected parser format. */
  std::cout << "Reading " << parser_string << " file ("
            << num_views << " cameras, "
            << num_features << " features)..." << std::endl;

  Bundle::Ptr bundle = Bundle::create();

  /* Read all cameras. */
  Bundle::Cameras& cameras = bundle->get_cameras();
  cameras.reserve(num_views);
  for (int i = 0; i < num_views; ++i)
  {
    cameras.push_back(CameraInfo());
    CameraInfo& cam = cameras.back();
    in >> cam.flen >> cam.dist[0] >> cam.dist[1];
    for (int j = 0; j < 9; ++j)
      in >> cam.rot[j];
    for (int j = 0; j < 3; ++j)
      in >> cam.trans[j];
  }

  if (in.eof())
    throw std::runtime_error("Unexpected EOF in bundle file");
  if (in.fail())
    throw std::runtime_error("Bundle file read error");

  /* Read all features. */
  Bundle::Features& features = bundle->get_features();
  features.reserve(num_features);
  for (int i = 0; i < num_features; ++i)
  {
    /* Insert the new (uninitialized) point. */
    features.push_back(Bundle::Feature3D());
    Bundle::Feature3D& feature = features.back();

    /* Read point position and color. */
    for (int j = 0; j < 3; ++j)
      in >> feature.pos[j];
    for (int j = 0; j < 3; ++j)
    {
      in >> feature.color[j];
      feature.color[j] /= 255.0f;
    }

    /* Read feature references. */
    int ref_amount = 0;
    in >> ref_amount;
    if (ref_amount < 0 || ref_amount > num_views)
    {
      in.close();
      throw std::runtime_error("Invalid feature reference amount");
    }

    for (int j = 0; j < ref_amount; ++j)
    {
      /*
       * Photosynther: The third parameter is the reprojection quality.
       * Bundler: The third and forth parameter are the floating point
       * x- and y-coordinate in an image-centered coordinate system.
       */
      Bundle::Feature2D ref;
      float dummy_float;
      if (format == BUNDLE_FORMAT_PHOTOSYNTHER)
      {
        in >> ref.view_id >> ref.feature_id;
        in >> dummy_float; // Drop reprojection quality.
        std::fill(ref.pos, ref.pos + 2, -1.0f);
      }
      else if (format == BUNDLE_FORMAT_NOAHBUNDLER)
      {
        in >> ref.view_id >> ref.feature_id;
        in >> ref.pos[0] >> ref.pos[1];
      }
      feature.refs.push_back(ref);
    }

    /* Check for premature EOF. */
    if (in.eof())
    {
      std::cerr << "Warning: Unexpected EOF (at feature "
                << i << ")" << std::endl;
      features.pop_back();
      break;
    }
  }

  in.close();
  return bundle;
}

/* ------------------ Support for Noah "Bundler"  ----------------- */

Bundle::Ptr
load_bundler_bundle (std::string const& filename)
{
  return load_bundler_ps_intern(filename, BUNDLE_FORMAT_NOAHBUNDLER);
}

/* ------------------- Support for Photosynther ------------------- */

Bundle::Ptr
load_photosynther_bundle (std::string const& filename)
{
  return load_bundler_ps_intern(filename, BUNDLE_FORMAT_PHOTOSYNTHER);
}

int main(int argc, char *argv[])
{
  ViewportList viewports;
  PairwiseMatching pairwise_matching;
  load_prebundle_from_file("/home/lucius/data/workspace/Ignatius_mve/view/prebundle.sfm", &viewports, &pairwise_matching);
}