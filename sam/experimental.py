import numpy as np
prompt = np.array([[1,0,1],
                   [0,0,1],
                   [1,1,0]])
print(f"prompt type is {prompt.shape}")
white_pixels = np.column_stack(np.where(prompt == 1))
print(f"white pixels type is {white_pixels.shape}")
print(f"white_pixels:\n{white_pixels}")
print(white_pixels.shape[0])
