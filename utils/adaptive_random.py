import random

class AdaptiveRandom:
  def __init__(self, values, weights=None):
    self.values = values
    self.weights = weights if weights else [1.0] * len(values)
    self.min_weight = 0.1  # Trọng số tối thiểu để không bị loại hoàn toàn
    self.decay = 0.7       # Hệ số giảm trọng số sau khi chọn

  def choose(self):
    total = sum(self.weights)
    probs = [w / total for w in self.weights]
    choice = random.choices(self.values, weights=probs, k=1)[0]

    # Giảm trọng số giá trị chọn
    index = self.values.index(choice)
    self.weights[index] = max(self.weights[index] * self.decay, self.min_weight)

    # Tăng nhẹ trọng số của các giá trị khác nhằm cần bằng lại
    for i in range(len(self.weights)):
        if i != index:
            self.weights[i] += (1 - self.weights[i]) * 0.1

    return choice