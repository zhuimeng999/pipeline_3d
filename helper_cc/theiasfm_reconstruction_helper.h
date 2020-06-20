//
// Created by lucius on 6/17/20.
//

#ifndef HELPER_CC_THEIASFM_RECONSTRUCTION_HELPER_H
#define HELPER_CC_THEIASFM_RECONSTRUCTION_HELPER_H

namespace math {
  typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector3f Vec3f;
  typedef Eigen::Matrix<unsigned char, 1, 3> Vec3uc;
}

namespace Sift {
class Options {

};

class Descriptors {

};
}
namespace Surf {
class Options {

};
class Descriptors {

};
}

struct CameraPose
{
  Eigen::Matrix3d K;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
};

/* -------------------- Common Data Structures -------------------- */
typedef std::pair<int, int> CorrespondenceIndex;
typedef std::vector<CorrespondenceIndex> CorrespondenceIndices;
/**
 * The FeatureSet holds per-feature information for a single view, and
 * allows to transparently compute and match multiple feature types.
 */
class FeatureSet
{
public:
  /** Bitmask with feature types. */
  enum FeatureTypes
  {
    FEATURE_SIFT = 1 << 0,
    FEATURE_SURF = 1 << 1,
    FEATURE_ALL = 0xFF
  };

  /** Options for feature detection and matching. */
  struct Options
  {
    Options (void);

    FeatureTypes feature_types;
    Sift::Options sift_opts;
    Surf::Options surf_opts;
  };

public:
  FeatureSet (void);
  explicit FeatureSet (Options const& options);
  void set_options (Options const& options);

  /** Computes the features specified in the options. */
  void compute_features (void* image);

  /** Normalizes the features positions w.r.t. the image dimensions. */
  void normalize_feature_positions (float px, float py);

  /** Clear descriptor data. */
  void clear_descriptors (void);

public:
  /** Image dimension used for feature computation. */
  int width, height;
  /** Per-feature image position. */
  std::vector<math::Vec2f> positions;
  /** Per-feature image color. */
  std::vector<math::Vec3uc> colors;
  /** The SIFT descriptors. */
  Sift::Descriptors sift_descriptors;
  /** The SURF descriptors. */
  Surf::Descriptors surf_descriptors;

private:
  void compute_sift (void * image);
  void compute_surf (void * image);

private:
  Options opts;
};

/* ------------------------ Implementation ------------------------ */

inline
FeatureSet::Options::Options (void)
    : feature_types(FEATURE_SIFT)
{
}

inline
FeatureSet::FeatureSet (void)
{
}

inline
FeatureSet::FeatureSet (Options const& options)
    : opts(options)
{
}

inline void
FeatureSet::set_options (Options const& options)
{
  this->opts = options;
}

/**
 * Per-viewport information.
 * Not all data is required for every step. It should be populated on demand
 * and cleaned as early as possible to keep memory consumption in bounds.
 */
struct Viewport
{
  /** Initial focal length estimate for the image. */
  float focal_length;
  /** Radial distortion parameter. */
  float radial_distortion[2];
  /** Principal point parameter. */
  float principal_point[2];

  /** Camera pose for the viewport. */
  CameraPose pose;

  /** The actual image data for debugging purposes. Usually nullptr! */
  void* image;
  /** Per-feature information. */
  FeatureSet features;
  /** Per-feature track ID, -1 if not part of a track. */
  std::vector<int> track_ids;
  /** Backup map from features to tracks that were removed due to errors. */
  std::unordered_map<int, int> backup_tracks;
};

/** The list of all viewports considered for bundling. */
typedef std::vector<Viewport> ViewportList;

/* --------------- Data Structure for Feature Tracks -------------- */

/** References a 2D feature in a specific view. */
struct FeatureReference
{
  FeatureReference (int view_id, int feature_id);

  int view_id;
  int feature_id;
};

/** The list of all feature references inside a track. */
typedef std::vector<FeatureReference> FeatureReferenceList;

/** Representation of a feature track. */
struct Track
{
  bool is_valid (void) const;
  void invalidate (void);
  void remove_view (int view_id);

  math::Vec3f pos;
  math::Vec3uc color;
  FeatureReferenceList features;
};

/** The list of all tracks. */
typedef std::vector<Track> TrackList;

/* Observation of a survey point in a specific view. */
struct SurveyObservation
{
  SurveyObservation (int view_id, float x, float y);

  int view_id;
  math::Vec2f pos;
};

/** The list of all survey point observations inside a survey point. */
typedef std::vector<SurveyObservation> SurveyObservationList;

/** Representation of a survey point. */
struct SurveyPoint
{
  math::Vec3f pos;
  SurveyObservationList observations;
};

/** The list of all survey poins. */
typedef std::vector<SurveyPoint> SurveyPointList;

/* ------------- Data Structures for Feature Matching ------------- */

/** The matching result between two views. */
struct TwoViewMatching
{
  bool operator< (TwoViewMatching const& rhs) const;

  int view_1_id;
  int view_2_id;
  CorrespondenceIndices matches;
};

/** The matching result between several pairs of views. */
typedef std::vector<TwoViewMatching> PairwiseMatching;

#endif //HELPER_CC_THEIASFM_RECONSTRUCTION_HELPER_H
