# Load necessary library
library(ggplot2)

# Update the dataset with the new data
data <- data.frame(
  Dataset = c("BUSI - Benign", "BUSI - Malignant", "BUSBRA", "BrEaST", "UDIAT - Benign", "UDIAT - Malignant", "STU", "STUDIED DATASET"),
  Dice_UNet = c(0.0879, 0.4286, 0.2166, 0.2214, 0.4613, 0.3991, 0.2898, 0.4827),  # Dice value for U-Net
  Dice_Single_Point = c(0.8063, 0.6462, 0.5143, 0.7016, 0.9074, 0.7823, 0.8215, 0.6656),  # Dice value for Single Point
  IoU_UNet = c(0.0578, 0.3457, 0.1447, 0.3903, 0.3289, 0.3457, 0.1983, 0.4340),  # IoU value for U-Net
  IoU_Single_Point = c(0.7215, 0.5180, 0.4281, 0.5967, 0.8434, 0.6679, 0.7345, 0.9206),  # IoU value for Single Point
  Dice_Single_Box = c(0.9105, 0.8681, 0.9147, 0.8965, 0.9406, 0.9110, 0.9146, 0.5795),  # Dice value for Single Box
  IoU_Single_Box = c(0.8421, 0.7718, 0.8456, 0.8159, 0.8891, 0.8383, 0.8443, 0.8531)  # IoU value for Single Box
)

# Visualize the updated table
print(data)

# Create the plot for IoU - Single Point vs U-Net
ggplot(data, aes(x = IoU_UNet, y = IoU_Single_Point, label = Dataset)) +
  geom_point(aes(color = Dataset), size = 4) +  # Add points with different colors for each dataset
  labs(x = "IoU - U-Net", y = "IoU - Single Point") +
  theme_minimal() +
  theme(text = element_text(size = 12)) +  # Font size
  scale_x_continuous(limits = c(0, 1)) +  # Adjust X axis to range from 0 to 1
  scale_y_continuous(limits = c(0, 1)) +  # Adjust Y axis to range from 0 to 1
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +  # Add a dashed diagonal line
  theme(legend.position = "right")  # Position the legend to the right of the plot

# Create the plot for Dice - Single Box vs U-Net
ggplot(data, aes(x = Dice_UNet, y = Dice_Single_Box, label = Dataset)) +
  geom_point(aes(color = Dataset), size = 4) +  # Add points with different colors for each dataset
  labs(x = "Dice Coefficient - U-Net", y = "Dice Coefficient - Single Box") +
  theme_minimal() +
  theme(text = element_text(size = 12)) +  # Font size
  scale_x_continuous(limits = c(0, 1)) +  # Adjust X axis to range from 0 to 1
  scale_y_continuous(limits = c(0, 1)) +  # Adjust Y axis to range from 0 to 1
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +  # Add a dashed diagonal line
  theme(legend.position = "right")  # Position the legend to the right of the plot

# Create the plot for IoU - Single Box vs U-Net
ggplot(data, aes(x = IoU_UNet, y = IoU_Single_Box, label = Dataset)) +
  geom_point(aes(color = Dataset), size = 4) +  # Add points with different colors for each dataset
  labs(x = "IoU - U-Net", y = "IoU - Single Box") +
  theme_minimal() +
  theme(text = element_text(size = 12)) +  # Font size
  scale_x_continuous(limits = c(0, 1)) +  # Adjust X axis to range from 0 to 1
  scale_y_continuous(limits = c(0, 1)) +  # Adjust Y axis to range from 0 to 1
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +  # Add a dashed diagonal line
  theme(legend.position = "right")  # Position the legend to the right of the plot

# Create the plot for Dice - Single Point vs U-Net (missing one)
ggplot(data, aes(x = Dice_UNet, y = Dice_Single_Point, label = Dataset)) +
  geom_point(aes(color = Dataset), size = 4) +  # Add points with different colors for each dataset
  labs(x = "Dice Coefficient - U-Net", y = "Dice Coefficient - Single Point") +
  theme_minimal() +
  theme(text = element_text(size = 12)) +  # Font size
  scale_x_continuous(limits = c(0, 1)) +  # Adjust X axis to range from 0 to 1
  scale_y_continuous(limits = c(0, 1)) +  # Adjust Y axis to range from 0 to 1
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +  # Add a dashed diagonal line
  theme(legend.position = "right")  # Position the legend to the right of the plot





# Create the grouped histogram
ggplot(data_long, aes(x = Dataset, y = Dice_Score, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +  # Grouped bars side by side
  scale_fill_manual(values = c("skyblue", "orange", "darkred")) +  # Colors for the models
  labs(title = "Dice Scores Across Datasets",
       x = "Datasets",
       y = "Dice Score",
       fill = "Model") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels
        plot.title = element_text(hjust = 0.5))  # Center the title



# Create the grouped histogram for the IoU Scores
ggplot(data_long_iou, aes(x = Dataset, y = IoU_Score, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +  # Grouped bars side by side
  scale_fill_manual(values = c("skyblue", "orange", "darkred")) +  # Colors for the models
  labs(title = "IoU Scores Across Datasets",
       x = "Datasets",
       y = "IoU Score",
       fill = "Model") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels
        plot.title = element_text(hjust = 0.5))  # Center the title
