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

#include "open_spiel/games/finite_crowd_modelling/finite_crowd_modelling.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <random>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/substitute.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace finite_crowd_modelling {
namespace {

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
                         /*long_name=*/"Finite Player Crowd Modelling",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/10,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {
                          {"players", GameParameter(kDefaultPlayers)},
                          {"size", GameParameter(kDefaultSize)},
                          {"horizon", GameParameter(kDefaultHorizon)},
                          {"init_pos_random", GameParameter(kInitRandomPos)},
                          {"target_move_prob", GameParameter(kTargetMoveProb)}},
                         /*default_loadable*/true,
                         /*provides_factored_observation_string*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  //std::cout << "FACTORY IN" << std::endl;
  auto a = std::shared_ptr<const Game>(new FiniteCrowdModellingGame(params));
  //std::cout << "FACTORY OUT" << std::endl;
  return a;
}

std::string StateToString(Player player, int position, int time) {
  return absl::Substitute("($0, $1, $2)", player, position, time);
  //Not sure if it'll ever reach here
  SpielFatalError(absl::Substitute(
      "Unexpected state - player_id: $0", player));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

FiniteCrowdModellingState::FiniteCrowdModellingState(std::shared_ptr<const Game> game,
                                         int num_players_, int size, int horizon, bool init_pos_random, double target_move_prob)
    : SimMoveState(game),
      num_players_(num_players_),
      size_(size),
      horizon_(horizon),
      init_pos_random_(init_pos_random),
      target_move_prob_(target_move_prob) {
        joint_action_.resize(num_players_, 0);
        player_positions_.resize(num_players_, 0);
        returns_.resize(num_players_, 0);
        emp_distribution_.resize(size_, 0);
        //std::cout << "Construct In" << std::endl;
        if (init_pos_random_) {
            std::random_device rd;
            std::mt19937 generator(rd());
            for (int p = 0; p < num_players_; ++p) {
              std::uniform_int_distribution<int> distribution(0, size_ - 1);
              player_positions_[p] = distribution(generator);
            }
        }
        //std::cout << "Construct Out" << std::endl;
      }

FiniteCrowdModellingState::FiniteCrowdModellingState(
    std::shared_ptr<const Game> game, int num_players_, int size, int horizon, 
    int t, bool init_pos_random, double target_move_prob)
    : SimMoveState(game),
      num_players_(num_players_),
      size_(size),
      horizon_(horizon),
      t_(t),
      init_pos_random_(init_pos_random),
      target_move_prob_(target_move_prob){
        joint_action_.resize(num_players_, 0);
        player_positions_.resize(num_players_, 0);
        returns_.resize(num_players_, 0);
        emp_distribution_.resize(size_, 0);
        //std::cout << "Construct IN" << std::endl;
        if (init_pos_random_) {
            std::random_device rd;
            std::mt19937 generator(rd());
            for (int p = 0; p < num_players_; ++p) {
              std::uniform_int_distribution<int> distribution(0, size_ - 1);
              player_positions_[p] = distribution(generator);
            }
        }
        //std::cout << "Construct Out" << std::endl;
      }

ActionsAndProbs FiniteCrowdModellingState::ChanceOutcomes() const {
  //NOT USED.
  ActionsAndProbs outcomes;
  //std::cout << "CHANCE IN" << std::endl;
  for (int i = 0; i < size_; ++i) {
    outcomes.push_back({i, 1. / size_});
  }
  //std::cout << "CHANCE OUT" << std::endl;
  return outcomes;
}

std::vector<Action> FiniteCrowdModellingState::LegalActions(Player player) const{
  if (IsTerminal()) {
    return {};
  }
  //Refers to indices of the kActionToMoveAt vector
  return {0, 1, 2};
}

void FiniteCrowdModellingState::DoApplyActions(const std::vector<Action>& actions) {
  SPIEL_CHECK_GE(actions.size(), 0);
  SPIEL_CHECK_FLOAT_EQ(actions.size(), num_players_);
  //std::cout << "DO APPLY ACTIONS START" << std::endl;
  joint_action_ = actions;

  // Seed the random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (auto p = Player{0}; p < num_players_; ++p) {
    // Generate a random number between 0 and 1
    double random_noise_value = dis(gen);

    SPIEL_CHECK_LE(p, num_players_);
    //std::cout<< player_positions_[p] << std::endl;
    //std::cout<< joint_action_[p] << std::endl;
    int updatedPosition = (player_positions_[p] + kActionToMove.at(joint_action_[p]) + size_) % size_;
    if (random_noise_value < noise_move_prob) {
      updatedPosition = (updatedPosition + 1 + size_) % size_;
    } 
    else if (random_noise_value < noise_move_prob + noise_move_prob) {
      updatedPosition = (updatedPosition - 1 + size_) % size_;
    }
    player_positions_[p] = updatedPosition;
  }
  //APPLY REWARDS
  AssignReturns();
  ++t_;
  //std::cout << "DO APPLY ACTIONS END" << std::endl;
}

std::string FiniteCrowdModellingState::ActionToString(Player player,
                                                Action action) const {
  //std::cout << "ATOS IN" << std::endl;
  std::string str = "";
  absl::StrAppend(&str, "P", player);
  absl::StrAppend(&str, " A: ",std::to_string(kActionToMove.at(action)));
  //std::cout << "ATOS OUT" << std::endl;
  return str;
}

void FiniteCrowdModellingState::EmpiricalDistribution() {
  //std::cout << "EMPIRIC DIST IN" << std::endl;
  for (auto p = Player{0}; p < num_players_; ++p) {
    //std::cout << "PLAYER POSITION" << player_positions_[p] << std::endl;
    emp_distribution_[player_positions_[p]]++;
  }
  //Normalize distribution
  for (int i = 0; i < size_; i++) {
    emp_distribution_[i] /= num_players_;
  }
}

bool FiniteCrowdModellingState::IsTerminal() const { return t_ >= horizon_; }

double FiniteCrowdModellingState::AssignRewards(Player current_player) const {
  //distance to center (the bar)
  //max reward: 1
  //std::cout << "ASSIGN REW IN" << std::endl;
  
  //std::cout<< "Player" << current_player << std::endl;
  double r_x = 1 - 1.0 * std::abs(player_positions_[current_player] - size_ / 2) / (size_ / 2);
  //std::cout << "R_X" << r_x << std::endl;
  //cost of action
  double r_a = -1.0 * std::abs(kActionToMove.at(joint_action_[current_player])) / size_;
  //std::cout << "R_A" << r_a << std::endl;
  //Natural log
  double r_mu = -std::log(emp_distribution_[player_positions_[current_player]]);
  //std::cout << "R_mu" << r_mu << std::endl;
  //std::cout << "ASSIGN REW OUT" << std::endl;
  return r_x + r_a + r_mu;
}

void FiniteCrowdModellingState::AssignReturns() {
  EmpiricalDistribution();
  //std::cout << "ASSIGN RETURNS IN" << std::endl;
  for (auto p = Player{0}; p < num_players_; ++p) {
    returns_[p] += AssignRewards(p);
    //std::cout << "RETURNS" << returns_[p] << std::endl;
  }
  //std::cout << "ASSIGN RETURNS OUT" << std::endl;
}

std::vector<double> FiniteCrowdModellingState::Returns() const {
  return returns_;
}

std::string FiniteCrowdModellingState::ToString(Player player) const {
  //std::cout << "TOSTRING WITH PARAM IN" << std::endl;
  auto a = StateToString(player, player_positions_[player], t_);
  //std::cout << "TOSTRING WITH PARAM OUT" << std::endl;
  
  return a;
}

std::string FiniteCrowdModellingState::ToString() const {
  std::string str = "";
  // //std::cout << "TOSTRING IN" << std::endl;
  // absl::StrAppend(&str, "Num_players", num_players_, horizon_, size_, "\n");
  // for (int p = 0; p < joint_action_.size(); ++p) {
  //   absl::StrAppend(&str, "P", p,
  //                   " action: ", ActionToString(p, joint_action_[p]), "\n");
  // }
  // //std::cout << "TOSTRING OUT" << std::endl;
  return str;
}

std::string FiniteCrowdModellingState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string FiniteCrowdModellingState::ObservationString(Player player) const {
  //std::cout << "OBS IN" << std::endl;
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  //std::cout << "OBS OUT" << std::endl;
  return ToString(player);
}

void FiniteCrowdModellingState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  //std::cout << "OBSTENS IN" << std::endl;
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
  //std::cout << "OBSTENS OUT" << std::endl;
}

std::unique_ptr<State> FiniteCrowdModellingState::Clone() const {
  //std::cout << "CLONE IN" << std::endl;
  auto a = std::unique_ptr<State>(new FiniteCrowdModellingState(*this));
  //std::cout << "CLONE OUT" << std::endl;
  return a;
}

FiniteCrowdModellingGame::FiniteCrowdModellingGame(const GameParameters& params)
    : Game(kGameType, params),
      size_(ParameterValue<int>("size", kDefaultSize)),
      num_players_(ParameterValue<int>("players")),
      horizon_(ParameterValue<int>("horizon", kDefaultHorizon)),
      init_pos_random_(ParameterValue<bool>("init_pos_random", kInitRandomPos)),
      target_move_prob_(ParameterValue<double>("target_move_prob", kTargetMoveProb)) {
        SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
        SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
      }

std::vector<int> FiniteCrowdModellingGame::ObservationTensorShape() const {
  // +1 to allow for t_ == horizon.
  return {size_ + horizon_ + 1};
}

}  // namespace finite_crowd_modelling
}  // namespace open_spiel

