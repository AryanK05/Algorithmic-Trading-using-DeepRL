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