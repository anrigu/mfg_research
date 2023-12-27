// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Mean Field Crowd Modelling Game.
//
// This game corresponds to the "Beach Bar Process" defined in section 4.2 of
// "Fictitious play for mean field games: Continuous time analysis and
// applications", Perrin & al. 2019 (https://arxiv.org/abs/2007.03458).
//
// In a nutshell, each representative agent evolves on a circle, with {left,
// neutral, right} actions. The reward includes the proximity to an imagined bar
// placed at a fixed location in the circle, and penalties for moving and for
// being in a crowded place.

#ifndef OPEN_SPIEL_GAMES_FINITE_CROWD_MODELLING_H_
#define OPEN_SPIEL_GAMES_FINITE_CROWD_MODELLING_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <random>

#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace finite_crowd_modelling {

inline constexpr int kNumPlayers = 5;
inline constexpr int kDefaultHorizon = 10;
inline constexpr int kDefaultSize = 10;
inline constexpr int kNumActions = 3;
inline constexpr int kNumChanceActions = 3;
inline constexpr int kDefaultPlayers = 2;
inline constexpr bool kInitRandomPos = false;
inline constexpr double kTargetMoveProb = 1;
// Action that leads to no displacement on the circle of the game.
inline constexpr int kNeutralAction = 1;

// Game state.
// The high-level state transitions are as follows:
// - First game state is a chance node where the initial position on the
//   circle is selected.
// Then we cycle over:
// 1. Decision node with actions {left, neutral, right}, represented by integers
//    0, 1, 2. This moves the position on the circle.
// 2. Mean field node, where we expect that external logic will call
//    DistributionSupport() and UpdateDistribution().
// 3. Chance node, where one of {left, neutral, right} actions is externally
//    selected.
// The game stops after a non-initial chance node when the horizon is reached.
class FiniteCrowdModellingState : public SimMoveState {
 public:
  FiniteCrowdModellingState(std::shared_ptr<const Game> game, int num_players_, int size, int horizon, 
                      bool init_pos_random, double target_move_prob);
  FiniteCrowdModellingState(std::shared_ptr<const Game> game, int num_players_, int size, int horizon,
                      int t, bool init_pos_random, double target_move_prob);

  FiniteCrowdModellingState(const FiniteCrowdModellingState&) = default;

  std::string ToString() const override;
  FiniteCrowdModellingState& operator=(const FiniteCrowdModellingState&) = default;

  std::string ActionToString(Player player, Action action) const override;
  std::string ToString(Player player) const;
  bool IsTerminal() const override;
  double AssignRewards(Player current_player) const;
  void AssignReturns();
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions(Player player) const override;
  ActionsAndProbs ChanceOutcomes() const override;
  void EmpiricalDistribution();
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : kSimultaneousPlayerId;
  }

 protected:
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  const int num_players_ = -1;
  // Size of the circle.
  const int size_ = -1;
  // Max time step
  const int horizon_ = -1;
  //Prob that action not affected by noise
  const double target_move_prob_ = 1;
  const double noise_move_prob = (1-target_move_prob_) / 2;
  const bool init_pos_random_;
  // Current time, in [0, horizon_].
  int t_ = 0;
  std::vector<double> emp_distribution_;
  std::vector<double> returns_;
  
  std::vector<Action> joint_action_;  // The action taken by all the players.
  std::vector<int> player_positions_;

  // kActionToMove[action] is the displacement on the circle of the game for
  // 'action'.
  static constexpr std::array<int, 3> kActionToMove = {-1, 0, 1};

};

class FiniteCrowdModellingGame : public Game {
 public:
  explicit FiniteCrowdModellingGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<FiniteCrowdModellingState>(shared_from_this(), num_players_, size_,
                                                  horizon_, init_pos_random_, target_move_prob_);
  }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override {
    return -std::numeric_limits<double>::infinity();
  }
  double MaxUtility() const override {
    return std::numeric_limits<double>::infinity();
  }
  int MaxGameLength() const override { return horizon_; }
  int MaxChanceNodesInHistory() const override {
    // + 1 to account for the initial extra chance node.
    return horizon_ + 1;
  }
  

  std::vector<int> ObservationTensorShape() const override;
  // int MaxChanceOutcomes() const override {
  //   return std::max(size_, kNumChanceActions);
  // }

 private:
  const int size_;
  const int horizon_;
  const int num_players_;
  const bool init_pos_random_;
  const double target_move_prob_;
};

}  // namespace finite_crowd_modelling
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_FINITE_CROWD_MODELLING_H_
