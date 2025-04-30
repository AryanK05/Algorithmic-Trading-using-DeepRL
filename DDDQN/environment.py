import numpy as np
import pandas as pd

class TradingEnvironment:
    def __init__(self, data, history_t=90, transaction_cost=0.001):
        self.data = data
        self.history_t = history_t
        self.transaction_cost = transaction_cost
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history_price = [0 for _ in range(self.history_t)]
        self.history_returns = [0 for _ in range(self.history_t)]
        return self._get_observation()
    
    def _get_observation(self):
        current_row = self.data.iloc[self.t, :]
        obs = [self.position_value, len(self.positions)]
        obs.extend(self.history_price)
        obs.extend(self.history_returns)
        obs.extend([
            current_row['Norm_Close'],  # Normalized close price
            current_row['SMA7'],        # 7-day Simple Moving Average
            current_row['SMA25'],       # 25-day Simple Moving Average
            current_row['EMA12'],       # 12-day Exponential Moving Average
            current_row['EMA26'],       # 26-day Exponential Moving Average
            current_row['MACD'],        # Moving Average Convergence Divergence
            current_row['Signal'],      # Signal line
            current_row['RSI'],         # Relative Strength Index
            current_row['Daily_Return'] # Daily return
        ])
        return obs
    
    def step(self, act):
        current_price = self.data.iloc[self.t, :]['Close']
        reward = 0
        info = {}

        if act == 1:  # Buy
            self.positions.append(current_price)
            cost = current_price * self.transaction_cost
            reward -= cost
            info.update(action='buy', price=current_price, cost=cost)

        elif act == 2:  # Sell
            if not self.positions:
                reward = -1
                info['action'] = 'sell_error'
            else:
                profits = 0
                for p in self.positions:
                    transaction_profit = (current_price - p) - (current_price * self.transaction_cost)
                    profits += transaction_profit
                reward = profits
                self.profits += profits
                self.positions.clear()
                info.update(action='sell', price=current_price, profit=profits)

        else:
            info['action'] = 'hold'

        # Increasing the time
        self.t += 1
        if self.t >= len(self.data) - 1:
            self.done = True

        # Update history if not done
        if not self.done:
            next_price = self.data.iloc[self.t, :]['Close']
            # update position value
            self.position_value = sum((next_price - p) for p in self.positions)
            # update histories
            self.history_price.pop(0)
            self.history_price.append(next_price)
            self.history_returns.pop(0)
            self.history_returns.append(self.data.iloc[self.t, :]['Daily_Return'])

        # Scaling rewards
        if not self.done:
            if reward > 0:
                reward = min(1.0, reward / current_price * 10)
            elif reward < 0:
                reward = max(-1.0, reward / current_price * 10)
        else:
            # Episode end liquidation uses current_price
            if self.positions:
                final_value = sum(
                    (current_price - p) - (current_price * self.transaction_cost)
                    for p in self.positions
                )
                reward += final_value
                info['final_liquidation'] = final_value
                self.positions.clear()
            info['total_profit'] = self.profits

        return self._get_observation(), reward, self.done, info
    



class DayTradingEnv:
    """
    A day trading environment which takes into account past history_length time steps into consideration

    The environment takes 2 pandas DataFrame, one with normalized price columns, other without normalization (to compute profit)
    with at least the following columns:
      - 'open': Opening prices
      - 'close': Closing prices
      - Any number of additional indicator columns

    Observations are the past history_length days of data for all features.

    Can be output in 2 ways:
      - '1D': a vector containing data from all days concatenated end to end
      - '2D': a matrix where one dimension is for the number of days, while the other is for features per day


    Actions:
      0 - Buy at opening price, sell at closing price
      1 - Sell at opening price, buy at closing price

    Reward:
      +1 for profitable trade, -1 for unprofitable trade or break-even

    Tracks:
      - total_profit: cumulative sum of profits
      - confusion_matrix: 2x2 matrix for actions taken vs optimal actions
    """
    def __init__(self, df: pd.DataFrame, non_normal_df: pd.DataFrame, history_length: int = 90, observation_dim='1D'):
        # Original data frame
        self.raw_df = df.reset_index(drop=True)
        self.history_length = history_length
        self.n_steps = len(self.raw_df)
        self.obs_dim = observation_dim

        
        self.non_normal = non_normal_df
        df = self.raw_df.copy()

        self.data = df
        self.feature_cols = df.columns.tolist()

        # Spaces
        self.action_space = 2
        self.observation_space = (history_length, len(self.feature_cols))

        # Trackers
        self.total_profit = 0.0
        self.invested_amount = 0.0
        self.steps = 0
        self.confusion_matrix = np.zeros((2, 2), dtype=int)

        # Internal state
        self.current_step = None

    def reset(self):
        """
        Reset the environment state and trackers.
        Starts at the earliest valid index (history_length).
        """
        self.current_step = self.history_length
        self.total_profit = 0.0
        self.invested_ammount = 0.0
        self.confusion_matrix = np.zeros((2, 2), dtype=int)
        return self._get_observation()

    def step(self, action: int):
        """
        Run one timestep of the environment's dynamics.
        Returns: observation, reward, done, info
        """
        assert action in [0, 1], "Invalid Action"
        self.steps += 1
        # Today's prices
        today = self.data.iloc[self.current_step]
        open_price, close_price = today['Open'], today['Close']

        true_data = self.non_normal.iloc[self.current_step]
        tr_open_price, tr_close_price = true_data['Open'], true_data['Close']

        

        # Profit calculation
        profit_normal = (close_price - open_price)/open_price if action == 0 else (open_price - close_price)/close_price
        reward = +1 if profit_normal > 0 else -1

        true_profit = (tr_close_price - tr_open_price) if action == 0 else (tr_open_price - tr_close_price)
        invested = tr_open_price if action == 0 else tr_close_price
        
        # Update trackers
        self.total_profit += true_profit
        self.invested_amount += invested
        row = 0 if profit_normal > 0 else 1
        col = action
        self.confusion_matrix[row, col] += 1

        # Advance
        self.current_step += 1
        done = self.current_step >= self.n_steps

        if not done:
            obs = self._get_observation()
        else:
            self.current_step -= 1
            obs = np.zeros_like(self._get_observation())
            self.current_step += 1
        info = {
            'open': open_price,
            'close': close_price,
            'profit': profit_normal,
            'total_profit': self.total_profit,
            'confusion_matrix': self.confusion_matrix.copy()
        }

        return obs, reward, done, info
    


    def _get_observation(self):
        """
        Get the past history_length days of data as the observation.
        """
        start = self.current_step - self.history_length
        end = self.current_step
        obs_df = self.data.iloc[start:end]

        assert self.obs_dim in ['1D', '2D'], "Invalid dim shape"
        
        # ===================
        if self.obs_dim == '1D':
            out = obs_df.values.astype(np.float32).flatten()
        elif self.obs_dim == '2D':
            out = obs_df.values.astype(np.float32)
        # ===================

        
        
        return out

    def render_confusion_matrix(self):
        """
        Returns the confusion matrix in a pandas DataFrame for readability:
             action=0  action=1
        pos         x11       x12
        neg         x21       x22
        """

        
        # self.confusion_matrix = self.confusion_matrix/self.steps * 100 # If percentages needed
        
        print( pd.DataFrame(
            self.confusion_matrix,
            index=['profit>=0', 'profit<0'],
            columns=['action=0_buy', 'action=1_sell']
        ))