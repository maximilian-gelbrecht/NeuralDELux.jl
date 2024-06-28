# NeuralDELux

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://maximilian-gelbrecht.github.io/NeuralDELux.jl/dev/)
[![Build Status](https://github.com/maximilian-gelbrecht/NeuralDELux.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/maximilian-gelbrecht/NeuralDELux.jl/actions/workflows/CI.yml?query=branch%3Amain)

Work in progress. Tools and Solvers for fast Neural DEs with Lux.jl. The solvers implemented here just iterate the system by one step in time. They are differentiable by Zygote and GPU compatible. Additionally there are tools for using the regular SciML solvers as well. 

The package mainly introduces two types: `ADNeuralDE` for the use with one step ahead solvers that can be differentiated efficiently by Zygote and `SciMLNeuralDE` for the use with SciML. 
