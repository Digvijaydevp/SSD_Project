import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const TrainingLineGraph = ({ trainingInfo }) => {
  // Add debugging console logs
  console.log("Training Info Received:", trainingInfo);
  console.log("Training Info Type:", typeof trainingInfo);
  console.log("Is Array:", Array.isArray(trainingInfo));

  // Check if trainingInfo is valid
  if (
    !trainingInfo ||
    !Array.isArray(trainingInfo) ||
    trainingInfo.length === 0
  ) {
    return (
      <div
        style={{
          backgroundColor: "#f8f9fa",
          borderRadius: "12px",
          padding: "20px",
          boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
          marginTop: "20px",
          textAlign: "center",
          color: "#6c757d",
        }}
      >
        <h3>No Training Data Available</h3>
        <p>Unable to render training performance graph.</p>
      </div>
    );
  }

  // Prepare data for the line chart
  const chartData = trainingInfo.map((epoch) => ({
    epoch: epoch.epoch,
    "Train Loss": epoch.train_loss,
    "Validation Loss": epoch.val_loss,
    "Validation Accuracy": epoch.val_accuracy * 100, // Convert to percentage
  }));

  console.log("Chart Data Prepared:", chartData);

  return (
    <div
      style={{
        backgroundColor: "#f8f9fa",
        borderRadius: "12px",
        padding: "20px",
        boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
        marginTop: "20px",
      }}
    >
      <h3
        style={{
          borderBottom: "2px solid #17a2b8",
          paddingBottom: "10px",
          marginBottom: "15px",
          color: "#17a2b8",
        }}
      >
        Training Performance Visualization
      </h3>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={chartData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="epoch"
            label={{
              value: "Epochs",
              position: "insideBottomRight",
              offset: -10,
            }}
          />
          <YAxis
            yAxisId="left"
            label={{
              value: "Loss",
              angle: -90,
              position: "insideLeft",
            }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            label={{
              value: "Accuracy (%)",
              angle: 90,
              position: "insideRight",
            }}
          />
          <Tooltip
            labelStyle={{ color: "#333" }}
            contentStyle={{ backgroundColor: "rgba(255,255,255,0.9)" }}
          />
          <Legend />

          <Line
            yAxisId="left"
            type="monotone"
            dataKey="Train Loss"
            stroke="#8884d8"
            activeDot={{ r: 8 }}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="Validation Loss"
            stroke="#82ca9d"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="Validation Accuracy"
            stroke="#ffc658"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TrainingLineGraph;
