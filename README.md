# ScenarioFuzz-LLM

**Enhancing Diversity in Autonomous Driving Scenario Fuzzing with Large Language Models (LLMs)**

---

## Introduction

ScenarioFuzz-LLM is an innovative framework developed to improve the safety testing of Autonomous Driving Systems (ADS) by enhancing scenario diversity. As ADS technologies become more prevalent in real-world applications, ensuring their reliability in rare and complex situations is crucial. However, traditional testing methods often encounter challenges in discovering diverse edge cases, which are essential for identifying new types of defects.

This project introduces ScenarioFuzz-LLM, a method that leverages Large Language Models (LLMs) to guide a genetic algorithm-based testing framework, aimed at breaking through diversity bottlenecks. By doing so, ScenarioFuzz-LLM facilitates a broader exploration of possible edge cases, enabling ADS testing to be more comprehensive and uncover a higher number of unique defects.

Our experiments demonstrate a 35.62% improvement in scenario diversity using ScenarioFuzz-LLM, which outperforms current state-of-the-art methods. This framework has successfully identified 24 unique defects in ADS, showcasing its efficacy in advancing ADS testing and improving overall system safety.

## Key Features

- **LLM-Guided Mutation**: ScenarioFuzz-LLM incorporates LLMs as expert agents to guide mutations when the genetic algorithm encounters stagnation, enhancing the diversity of testing scenarios.
- **Multi-Objective Optimization**: Evaluates scenarios based on metrics such as minimum vehicle distance, time-to-collision, and scenario variability to generate meaningful and diverse test cases.
- **Broad Edge Case Coverage**: Allows the testing framework to explore a wide array of potential ADS failures by continuously adapting and evolving test scenarios.
- **Integration with CARLA Simulator**: Provides a comprehensive testing setup for ADS simulation using CARLA, making ScenarioFuzz-LLM compatible with the Autoware.ai platform.

## Repository Overview

This repository includes the following components:
- **Source Code**: Implementing the ScenarioFuzz-LLM framework and its integration with CARLA.
- **Simulation Scripts**: Scripts to set up and run tests on CARLA, including scenario generation, mutation processes, and defect logging.
- **Pre-trained Models and Prompts**: Optimized prompts and models for guided scenario mutation and diversity evaluation.
- **Data and Results**: Dataset for initial test cases, along with results and statistics of our experiments, demonstrating the effectiveness of ScenarioFuzz-LLM.

## Getting Started

To get started, follow these instructions to set up the environment and run your first scenario tests.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ldegao/ScenarioFuzz-LLM.git



## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

