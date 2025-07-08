{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils}: 
    flake-utils.lib.eachDefaultSystem (system: 
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
    {
      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python313
          python313Packages.pip
          python313Packages.pandas
          python313Packages.openpyxl
          python313Packages.opencv4
          python313Packages.torchvision
          python313Packages.torch
          # python313Packages.numpy
          # python313Packages.scikit-learn
          # python313Packages.matplotlib

        ];
      };
    });
}