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

#include "open_spiel/games/crowd_modelling_finite/crowd_modelling.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/substitute.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace crowd_modelling {
namespace {
inline constexpr float kEpsilon = 1e-25;


// // Player ids are 0, 1, 2, ...
// // Negative numbers are used for various special values.
// enum PlayerId {
//   // Player 0 is always valid, and is used in single-player games.
//   kDefaultPlayerId = 0,
//   // The fixed player id for chance/nature.
//   kChancePlayerId = -1,
//   // What is returned as a player id when the game is simultaneous.
//   kSimultaneousPlayerId = -2,
//   // Invalid player.
//   kInvalidPlayer = -3,
//   // What is returned as the player id on terminal nodes.
//   kTerminalPlayerId = -4,
//   // player id of a mean field node
//   kMeanFieldPlayerId = -5
// };


// Facts about the game.
const GameType kGameType{/*short_name=*/"finite_crowd_modelling",
                         /*long_name=*/"Finite Game Crowd Modelling",
                         GameType::Dynamics::kExplicitStochastic,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=kNumPlayers*/,
                         min_num_players=1,
                         num_players_=kNumPlayers,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"size", GameParameter(kDefaultSize)},
                          {"horizon", GameParameter(kDefaultHorizon)}},
                         /*default_loadable*/true,
                         /*provides_factored_observation_string*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CrowdModellingGame(params));
}

std::string StateToString(Player player) {
  return absl::Substitute(player, player_positions_[player], t_);
  //Not sure if it'll ever reach here
  SpielFatalError(absl::Substitute(
      "Unexpected state - player_id: ", player);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

CrowdModellingState::CrowdModellingState(std::shared_ptr<const Game> game,
                                         int size, int horizon)
    : State(game),
      size_(size),
      horizon_(horizon) {
        joint_action_.resize(num_players_, 0);
        player_positions_.resize(num_players_, 0);
        returns_.resize(num_players_, 0);
      }

CrowdModellingState::CrowdModellingState(
    std::shared_ptr<const Game> game, int size, int horizon, 
    int t)
    : State(game),
      size_(size),
      horizon_(horizon),
      t_(t){
        joint_action_.resize(num_players_, 0);
        player_positions_.resize(num_players_, 0);
        returns_.resize(num_players_, 0);
      }

//Come back to look at this
std::vector<Action> CrowdModellingState::LegalActions() const {
  if (IsTerminal()) return {};
  // if (IsChanceNode()) return LegalChanceOutcomes();
  SPIEL_CHECK_TRUE(IsPlayerNode());
  return {0, 1, 2};
}

ActionsAndProbs CrowdModellingState::ChanceOutcomes() const {
  if (is_chance_init_) {
    ActionsAndProbs outcomes;
    for (int i = 0; i < size_; ++i) {
      outcomes.push_back({i, 1. / size_});
    }
    return outcomes;
  }
  return {{0, 1. / 3}, {1, 1. / 3}, {2, 1. / 3}};
}

void CrowdModellingState::DoApplyAction(const std::vector<Action>& actions) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, size_);

  joint_action_ = actions;

  for (auto p = Player{0}; p < num_players_; ++p) {
    SPIEL_CHECK_LE(p, num_players_);
    int updatedPosition = (player_positions_[p] + kActionToMove.at(joint_action_[p]) + size_) % size_;
    player_positions_[p] = updatedPosition;
  }
  ++t_;
}

std::string CrowdModellingState::ActionToString(Player player,
                                                Action action) const {
  return std::to_string(kActionToMove.at(action));
}

std::vector<double> CrowdModellingState::EmpiricalDistribution() {
  std::vector<double> emp_distribution_(size_, 0.);
  for (auto p = Player{0}; p < num_players_; ++p) {
    emp_distribution_[player_positions_[p]]++;
  }
  //Normalize distribution
  for (int i = 0; i < size_; i++) {
    emp_distribution_[i] / num_players_;
  }
  return emp_distribution_;
}

bool CrowdModellingState::IsTerminal() const { return t_ >= horizon_; }

double CrowdModellingState::AssignRewards(Player current_player) const {
  //distance to center (the bar)
  //max reward: 1
  double r_x = 1 - 1.0 * std::abs(player_positions_[current_player] - size_ / 2) / (size_ / 2);
  //cost of action
  double r_a = -1.0 * std::abs(kActionToMove.at(joint_action_[current_player])) / size_;
  double r_mu = -std::log(distribution_[player_positions_[current_player]]+kEpsilon);
  return r_x + r_a + r_mu;
}

void CrowdModellingState::AssignReturns() {
  std::vector<int> distribution_ = EmpiricalDistribution();
  for (auto p = Player{0}; p < num_players_; ++p) {
    returns_[p] += Rewards(p);
  }
}

std::vector<double> CrowdModellingState::Returns() const {
  return returns_;
}

std::string CrowdModellingState::ToString(Player player) const {
  return StateToString(player_positions_[player], t_, player);
}

std::string CrowdModellingState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string CrowdModellingState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString(player);
}

void CrowdModellingState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  //size of value is num actions + num time steps + 1 (for initial t = 0)
  SPIEL_CHECK_EQ(values.size(), size_ + horizon_ + 1);
  SPIEL_CHECK_LT(player_positions_[player], size_);
  SPIEL_CHECK_GE(t_, 0);
  // Allow t_ == horizon_.
  SPIEL_CHECK_LE(t_, horizon_);
  std::fill(values.begin(), values.end(), 0.);
  if (player_positions_[player] >= 0) {
    values[player_positions_[player]] = 1.;
  }
  // x_ equals -1 for the initial (blank) state, don't set any
  // position bit in that case.
  // Current timestep
  values[size_ + t_] = 1.;
}

std::unique_ptr<State> CrowdModellingState::Clone() const {
  return std::unique_ptr<State>(new CrowdModellingState(*this));
}

CrowdModellingGame::CrowdModellingGame(const GameParameters& params)
    : Game(kGameType, params),
      size_(ParameterValue<int>("size", kDefaultSize)),
      horizon_(ParameterValue<int>("horizon", kDefaultHorizon)) {}

std::vector<int> CrowdModellingGame::ObservationTensorShape() const {
  // +1 to allow for t_ == horizon.
  return {size_ + horizon_ + 1};
}

}  // namespace crowd_modelling
}  // namespace open_spiel
