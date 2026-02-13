from assignment2 import mlb_position_player_salary

model, val_loss = mlb_position_player_salary("baseball.txt")

print(model)
print("Validation loss:", val_loss)
