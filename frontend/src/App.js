import React, { useState, useEffect } from "react";
import TrainingLineGraph from './TrainingLineGraph';


function App() {
  const [inputValue, setInputValue] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysisMode, setAnalysisMode] = useState("");
  const [currentView, setCurrentView] = useState("home"); // "home", "analysis", "reports"
  const [reports, setReports] = useState(null); // Stores report data
  const [activeTab, setActiveTab] = useState("positive");

  useEffect(() => {
    if (reports) {
      console.log("Received Reports:", reports);
    }
  }, [reports]);

  // Handles analysis for hashtags or custom text
  const handleAnalyze = async () => {
    setLoading(true);
    const url =
      analysisMode === "search"
        ? "http://127.0.0.1:5000/hashtag"
        : "http://127.0.0.1:5000/custom-text";
    const key = analysisMode === "search" ? "input" : "text";

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ [key]: inputValue }),
    });
    const data = await response.json();
    setResults(data);
    setLoading(false);
  };

  const fetchReports = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/reports");

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      setReports(data);
    } catch (error) {
      console.error("Failed to fetch reports:", error);
      // Optionally set an error state
      setReports(null);
    }
  };

  return (
    <div style={styles.container}>
      {/* Home Screen */}
      {currentView === "home" && (
        <div style={styles.homeContainer}>
          <h1 style={styles.title}>Sentiment Analyzer</h1>
          <div style={styles.optionsContainer}>
            <button
              style={styles.optionButton}
              onClick={() => setCurrentView("analysis")}
            >
              Analyze Sentiments
            </button>
            <button
              style={styles.optionButton}
              onClick={async () => {
                await fetchReports();
                setCurrentView("reports");
              }}
            >
              View Reports
            </button>
          </div>
        </div>
      )}

      {/* Analysis Screen */}
      {currentView === "analysis" && (
        <div style={styles.resultsContainer}>
          {!results ? (
            <>
              <h2 style={styles.resultsTitle}>Analyze Sentiments</h2>
              <div style={styles.analysisOptions}>
                <button
                  style={styles.optionButton}
                  onClick={() => setAnalysisMode("search")}
                >
                  Search Keywords/Hashtags
                </button>
                <button
                  style={styles.optionButton}
                  onClick={() => setAnalysisMode("custom")}
                >
                  Analyze Custom Text
                </button>
              </div>
              {analysisMode && (
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleAnalyze();
                  }}
                  style={styles.searchForm}
                >
                  <input
                    type="text"
                    placeholder={
                      analysisMode === "search"
                        ? "Enter Keyword or Hashtag"
                        : "Enter Custom Text"
                    }
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    style={styles.searchInput}
                  />
                  <button type="submit" style={styles.analyzeButton}>
                    {loading ? "Analyzing..." : "Analyze"}
                  </button>
                </form>
              )}
            </>
          ) : (
            <div>
              {analysisMode === "search" && results && (
                <div>
                  <h2 style={styles.resultsTitle}>
                    Search Results for "{inputValue}"
                  </h2>

                  <div style={styles.overallSentimentContainer}>
                    <h3>
                      Overall Sentiment:
                      <span
                        style={{
                          ...styles.overallSentimentText,
                          ...styles[
                            results.overall_sentiment.sentiment.toLowerCase()
                          ],
                        }}
                      >
                        {results.overall_sentiment.sentiment} (
                        {results.overall_sentiment.percentage})
                      </span>
                    </h3>
                  </div>
                  <div style={styles.sentimentTabs}>
                    {["Positive", "Negative", "Neutral"].map((sentiment) => (
                      <button
                        key={sentiment}
                        style={{
                          ...styles.tabButton,
                          ...(activeTab === sentiment.toLowerCase()
                            ? styles.activeTab
                            : {}),
                        }}
                        onClick={() => setActiveTab(sentiment.toLowerCase())}
                      >
                        {sentiment} (
                        {results.tweets?.[sentiment.toLowerCase()]?.length || 0}
                        )
                      </button>
                    ))}
                  </div>

                  <table style={styles.table}>
                    <thead>
                      <tr>
                        <th style={styles.tableHeader}>Sentiment</th>
                        <th style={styles.tableHeader}>Tweet</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.tweets?.[activeTab]?.map((tweet, index) => (
                        <tr key={index} style={styles.tableRowHover}>
                          <td
                            style={{
                              ...styles.tableCellSentiment,
                              ...styles[tweet.sentiment.toLowerCase()],
                            }}
                          >
                            {tweet.sentiment.charAt(0).toUpperCase() +
                              tweet.sentiment.slice(1)}
                          </td>
                          <td style={styles.tableCell}>{tweet.text}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
               {analysisMode === "custom" && (
                <div>
                  <h2 style={styles.resultsTitle}>
                    Sentiment Analysis Results
                  </h2>
                  <p>Text: {results.text}</p>
                  <p>
                    Predicted Sentiment:{" "}
                    <span
                      style={styles[results.predicted_sentiment.toLowerCase()]}
                    >
                      {results.predicted_sentiment}
                    </span>
                  </p>
                </div>
              )}
            
            </div>
          )}
          
          <button
            onClick={() => {
              setResults(null);
              setInputValue("");
              setCurrentView("home");
            }}
            style={styles.homeButton}
          >
            Back to Home
          </button>
        </div>
      )}

      {/* Reports Screen */}
      {currentView === "reports" && (
        <div style={styles.resultsContainer}>
          <h2
            style={{
              textAlign: "center",
              color: "#333",
              marginBottom: "20px",
              fontSize: "24px",
              fontWeight: "bold",
            }}
          >
            Model Performance Report
          </h2>

          {reports ? (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "20px",
                maxWidth: "800px",
                margin: "0 auto",
              }}
            >
              {/* Performance Metrics Card */}
              <div
                style={{
                  backgroundColor: "#f8f9fa",
                  borderRadius: "12px",
                  padding: "20px",
                  boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
                }}
              >
                <h3
                  style={{
                    borderBottom: "2px solid #007bff",
                    paddingBottom: "10px",
                    marginBottom: "15px",
                    color: "#007bff",
                  }}
                >
                  Performance Metrics
                </h3>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "15px",
                  }}
                >
                  {Object.entries(reports.performance_metrics)
                    .filter(([key]) =>
                      ["accuracy", "f1_score", "precision", "recall"].includes(
                        key
                      )
                    )
                    .map(([metric, value]) => (
                      <div
                        key={metric}
                        style={{
                          backgroundColor: "#e9ecef",
                          borderRadius: "8px",
                          padding: "12px",
                          textAlign: "center",
                        }}
                      >
                        <div
                          style={{
                            fontSize: "14px",
                            color: "#495057",
                            marginBottom: "5px",
                            textTransform: "uppercase",
                          }}
                        >
                          {metric.replace("_", " ")}
                        </div>
                        <div
                          style={{
                            fontSize: "20px",
                            fontWeight: "bold",
                            color:
                              value > 0.7
                                ? "#28a745"
                                : value > 0.5
                                ? "#ffc107"
                                : "#dc3545",
                          }}
                        >
                          {(value * 100).toFixed(2)}%
                        </div>
                      </div>
                    ))}
                </div>
              </div>
              {/* Confusion Matrix Card */}
              <div
                style={{
                  backgroundColor: "#f8f9fa",
                  borderRadius: "12px",
                  padding: "20px",
                  boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
                }}
              >
                <h3
                  style={{
                    borderBottom: "2px solid #6f42c1",
                    paddingBottom: "10px",
                    marginBottom: "15px",
                    color: "#6f42c1",
                  }}
                >
                  Confusion Matrix
                </h3>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    overflowX: "auto",
                  }}
                >
                  <table
                    style={{
                      borderCollapse: "collapse",
                      width: "100%",
                      maxWidth: "400px",
                    }}
                  >
                    {reports.performance_metrics.confusion_matrix.map(
                      (row, rowIndex) => (
                        <tr key={rowIndex}>
                          {row.map((cell, cellIndex) => (
                            <td
                              key={cellIndex}
                              style={{
                                border: "1px solid #dee2e6",
                                padding: "10px",
                                textAlign: "center",
                                backgroundColor:
                                  cell > 0
                                    ? `rgba(40, 167, 69, ${Math.min(
                                        cell / 1000,
                                        0.7
                                      )})`
                                    : "#f8f9fa",
                              }}
                            >
                              {cell}
                            </td>
                          ))}
                        </tr>
                      )
                    )}
                  </table>
                </div>
              </div>
              {/* Class Distribution Card */}
              <div
                style={{
                  backgroundColor: "#f8f9fa",
                  borderRadius: "12px",
                  padding: "20px",
                  boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
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
                  Class Distribution
                </h3>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    gap: "15px",
                  }}
                >
                  {reports.class_distribution.map((cls, index) => (
                    <div
                      key={index}
                      style={{
                        backgroundColor: "#e9ecef",
                        borderRadius: "8px",
                        padding: "12px",
                        textAlign: "center",
                      }}
                    >
                      <div
                        style={{
                          fontSize: "14px",
                          color: "#495057",
                          marginBottom: "5px",
                        }}
                      >
                        {cls.name}
                      </div>
                      <div
                        style={{
                          fontSize: "20px",
                          fontWeight: "bold",
                          color: "#007bff",
                        }}
                      >
                        {cls.value}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {/* Training Information Card */}
              {reports.training_info && (
                <>
                  <TrainingLineGraph trainingInfo={reports.training_info} />
                  <div
                    style={{
                      backgroundColor: "#f8f9fa",
                      borderRadius: "12px",
                      padding: "20px",
                      boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
                    }}
                  >
                    <h3
                      style={{
                        borderBottom: "2px solid #28a745",
                        paddingBottom: "10px",
                        marginBottom: "15px",
                        color: "#28a745",
                      }}
                    >
                      Training Information
                    </h3>

                    {/* Debug Logging */}
                    {(() => {
                      return null;
                    })()}

                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr 1fr",
                        gap: "15px",
                      }}
                    >
                      {/* Total Epochs */}
                      <div
                        style={{
                          backgroundColor: "#e9ecef",
                          borderRadius: "8px",
                          padding: "12px",
                          textAlign: "center",
                        }}
                      >
                        <div
                          style={{
                            fontSize: "14px",
                            color: "#495057",
                            marginBottom: "5px",
                          }}
                        >
                          Total Epochs
                        </div>
                        <div
                          style={{
                            fontSize: "20px",
                            fontWeight: "bold",
                            color: "#007bff",
                          }}
                        >
                          {reports.training_info.length}
                        </div>
                      </div>

                      {/* Best Epoch (based on highest validation accuracy) */}
                      <div
                        style={{
                          backgroundColor: "#e9ecef",
                          borderRadius: "8px",
                          padding: "12px",
                          textAlign: "center",
                        }}
                      >
                        <div
                          style={{
                            fontSize: "14px",
                            color: "#495057",
                            marginBottom: "5px",
                          }}
                        >
                          Best Epoch
                        </div>
                        <div
                          style={{
                            fontSize: "20px",
                            fontWeight: "bold",
                            color: "#6f42c1",
                          }}
                        >
                          {
                            reports.training_info.reduce((best, current) =>
                              current.val_accuracy > best.val_accuracy
                                ? current
                                : best
                            ).epoch
                          }
                        </div>
                      </div>

                      {/* Best Validation Accuracy */}
                      <div
                        style={{
                          backgroundColor: "#e9ecef",
                          borderRadius: "8px",
                          padding: "12px",
                          textAlign: "center",
                        }}
                      >
                        <div
                          style={{
                            fontSize: "14px",
                            color: "#495057",
                            marginBottom: "5px",
                          }}
                        >
                          Best Validation Accuracy
                        </div>
                        <div
                          style={{
                            fontSize: "20px",
                            fontWeight: "bold",
                            color: "#28a745",
                          }}
                        >
                          {(
                            Math.max(
                              ...reports.training_info.map(
                                (e) => e.val_accuracy
                              )
                            ) * 100
                          ).toFixed(2)}
                          %
                        </div>
                      </div>
                    </div>

                    {/* Epoch-wise Performance Table */}
                    <div
                      style={{
                        marginTop: "20px",
                        overflowX: "auto",
                      }}
                    >
                      <h4
                        style={{
                          color: "#495057",
                          marginBottom: "10px",
                        }}
                      >
                        Epoch-wise Performance
                      </h4>
                      <table
                        style={{
                          width: "100%",
                          borderCollapse: "collapse",
                          minWidth: "600px",
                        }}
                      >
                        <thead>
                          <tr
                            style={{
                              backgroundColor: "#e9ecef",
                              borderBottom: "2px solid #dee2e6",
                            }}
                          >
                            <th style={tableHeaderStyle}>Epoch</th>
                            <th style={tableHeaderStyle}>Train Loss</th>
                            <th style={tableHeaderStyle}>Validation Loss</th>
                            <th style={tableHeaderStyle}>
                              Validation Accuracy
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {reports.training_info.map((epoch, index) => (
                            <tr
                              key={index}
                              style={{
                                backgroundColor:
                                  index % 2 === 0 ? "#f8f9fa" : "white",
                                borderBottom: "1px solid #dee2e6",
                              }}
                            >
                              <td style={tableCellStyle}>{epoch.epoch}</td>
                              <td style={tableCellStyle}>
                                {epoch.train_loss.toFixed(4)}
                              </td>
                              <td style={tableCellStyle}>
                                {epoch.val_loss.toFixed(4)}
                              </td>
                              <td style={tableCellStyle}>
                                {(epoch.val_accuracy * 100).toFixed(2)}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </div>
          ) : (
            <p style={{ textAlign: "center", color: "#6c757d" }}>
              Loading reports...
            </p>
          )}

          <div
            style={{
              display: "flex",
              justifyContent: "center",
              marginTop: "20px",
            }}
          >
            <button
              onClick={() => setCurrentView("home")}
              style={{
                backgroundColor: "#007bff",
                color: "white",
                border: "none",
                padding: "10px 20px",
                borderRadius: "5px",
                cursor: "pointer",
                transition: "background-color 0.3s ease",
              }}
              onMouseOver={(e) => (e.target.style.backgroundColor = "#0056b3")}
              onMouseOut={(e) => (e.target.style.backgroundColor = "#007bff")}
            >
              Back to Home
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

const tableHeaderStyle = {
  padding: "10px",
  textAlign: "center",
  color: "#495057",
  fontSize: "14px",
};

const tableCellStyle = {
  padding: "10px",
  textAlign: "center",
  fontSize: "14px",
};

const tableDataStyle = {
  padding: "10px",
  textAlign: "center",
  fontSize: "14px",
  color: "#495057", 
};

const styles = {
  container: {
    fontFamily: "'Helvetica Neue', Arial, sans-serif",
    padding: "20px",
    textAlign: "center",
    backgroundColor: "#f4f7fa",
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
  },
  homeContainer: {
    marginTop: "50px",
    maxWidth: "600px",
    padding: "20px",
    borderRadius: "8px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
    backgroundColor: "#ffffff",
  },
  analysisOptions: {
    display: "flex",
    justifyContent: "center",
    gap: "40px", 
    marginTop: "20px",
  },
  title: {
    fontSize: "36px",
    marginBottom: "20px",
    color: "#333",
  },
  overallSentimentContainer: {
    marginBottom: '15px',
    textAlign: 'center',
  },
  overallSentimentText: {
    marginLeft: '10px',
    fontWeight: 'bold',
  },
  optionsContainer: {
    display: "flex",
    justifyContent: "center",
    gap: "20px",
    marginTop: "20px",
  },
  optionButton: {
    padding: "12px 25px",
    backgroundColor: "#007bff",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    transition: "background-color 0.3s",
  },
  optionButtonHover: {
    backgroundColor: "#0056b3",
  },
  searchForm: {
    marginTop: "20px",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    gap: "10px",
  },
  searchInput: {
    padding: "12px",
    width: "250px",
    borderRadius: "5px",
    border: "1px solid #ddd",
  },
  analyzeButton: {
    padding: "12px 25px",
    backgroundColor: "#28a745",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    transition: "background-color 0.3s",
  },
  resultsContainer: {
    marginTop: "20px",
    overflowX: "auto", 
    width: "90%",
    maxWidth: "800px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
    backgroundColor: "#fff",
    borderRadius: "8px",
    padding: "20px",
  },
  sentimentTabs: {
    display: "flex",
    marginBottom: "20px",
    borderBottom: "1px solid #ddd",
  },
  tabButton: {
    padding: "10px 15px",
    border: "none",
    backgroundColor: "#f0f0f0",
    cursor: "pointer",
    marginRight: "5px",
  },
  activeTab: {
    backgroundColor: "#007bff",
    color: "white",
  },
  resultsTitle: {
    fontSize: "28px",
    marginBottom: "15px",
    color: "#333",
  },
  table: {
    margin: "0 auto",
    width: "100%",
    borderCollapse: "collapse",
    borderRadius: "8px",
    backgroundColor: "#f9f9f9",
  },
  tableHeader: {
    backgroundColor: "#007bff",
    color: "#fff",
    fontWeight: "bold",
    padding: "12px 20px",
    textAlign: "left",
  },
  tableCell: {
    padding: "12px 20px",
    border: "1px solid #ddd",
    textAlign: "left",
  },
  tableRowHover: {
    backgroundColor: "#f1f1f1", 
  },
  tableCellSentiment: {
    padding: "10px",
    border: "1px solid #ddd",
    color: "white", 
    fontWeight: "bold",
  },
  positive: {
    backgroundColor: "#28a745", // Green for positive sentiment
  },
  neutral: {
    backgroundColor: "#ffc107", // Yellow for neutral sentiment
  },
  negative: {
    backgroundColor: "#dc3545", // Red for negative sentiment
  },

  resultsTable: {
    width: "100%",
    borderCollapse: "collapse",
    marginTop: "20px",
  },

  resultsTableHeader: {
    backgroundColor: "#007bff",
    color: "#fff",
    fontWeight: "bold",
    padding: "12px",
    textAlign: "left",
  },

  resultsTableCell: {
    padding: "12px 20px",
    border: "1px solid #ddd",
    textAlign: "left",
  },

  positive: {
    backgroundColor: "#28a745", // Green for positive sentiment
    color: "#fff",
    fontWeight: "bold",
    padding: "6px 12px",
    borderRadius: "5px",
  },

  neutral: {
    backgroundColor: "#ffc107", // Yellow for neutral sentiment
    color: "#fff",
    fontWeight: "bold",
    padding: "6px 12px",
    borderRadius: "5px",
  },

  negative: {
    backgroundColor: "#dc3545", // Red for negative sentiment
    color: "#fff",
    fontWeight: "bold",
    padding: "6px 12px",
    borderRadius: "5px",
  },
  homeButton: {
    marginTop: "20px",
    padding: "12px 25px",
    backgroundColor: "#dc3545",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    transition: "background-color 0.3s",
  },
};



export default App;
