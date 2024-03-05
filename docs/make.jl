using Documenter, BilevelHeuristics


makedocs(
         format = Documenter.HTML(
                                  prettyurls = get(ENV, "CI", nothing) == "true",
                                  # assets = ["assets/favicon.ico"],
                                  # analytics = "UA-111111111-1",
                                 ),
         sitename="BilevelHeuristics.jl",
         authors = "Jesús Mejía",
         pages = [
                  "Index" => "index.md",
                  "Algorithms" => "algorithms.md",
                  "Examples" => "examples.md",
                  "Problems" => "problems.md",
                  "API References" => "api.md",
                 ]
        )



deploydocs(
           repo = "github.com/jmejia8/BilevelHeuristics.jl.git",
           devbranch = "main",
          )
