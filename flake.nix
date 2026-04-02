{
  description = "A Nix-flake-based Python development environment";
  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1";
  outputs =
    { ... }@inputs:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            inherit system;
            pkgs = import inputs.nixpkgs { inherit system; };
          }
        );
      version = "3.13";
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs, system }:
        let
          concatMajorMinor =
            v:
            pkgs.lib.pipe v [
              pkgs.lib.versions.splitVersion
              (pkgs.lib.sublist 0 2)
              pkgs.lib.concatStrings
            ];

          python = pkgs."python${concatMajorMinor version}";
        in
        {
          default = pkgs.mkShellNoCC {
            venvDir = ".venv";
            postShellHook = ''
              venvVersionWarn() {
              	local venvVersion
              	venvVersion="$("$venvDir/bin/python" -c 'import platform; print(platform.python_version())')"

              	[[ "$venvVersion" == "${python.version}" ]] && return

              	cat <<EOF
              Warning: Python version mismatch: [$venvVersion (venv)] != [${python.version}]
                       Delete '$venvDir' and reload to rebuild for version ${python.version}
              EOF
              }

              venvVersionWarn
            '';
            packages = (
              with python.pkgs;
              [
                venvShellHook
                pip
              ]
            );
          };
        }
      );
    };
}
