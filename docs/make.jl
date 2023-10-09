using NeuralDELux
using Documenter

DocMeta.setdocmeta!(NeuralDELux, :DocTestSetup, :(using NeuralDELux); recursive=true)

makedocs(;
    modules=[NeuralDELux],
    authors="Maximilian Gelbrecht <maximilian.gelbrecht@posteo.de> and contributors",
    repo="https://github.com/maximilian-gelbrecht/NeuralDELux.jl/blob/{commit}{path}#{line}",
    sitename="NeuralDELux.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://maximilian-gelbrecht.github.io/NeuralDELux.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/maximilian-gelbrecht/NeuralDELux.jl",
    devbranch="main",
)
