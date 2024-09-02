import toml

# Load existing pyproject.toml
with open("pyproject.toml", "r", encoding="utf-8") as file:
    pyproject = toml.load(file)

# Read requirements.txt and add to dependencies
with open("requirements/requirements.txt", "r", encoding="utf-8") as req_file:
    requirements = req_file.read().splitlines()

# Ensure the dependencies section exists
if "project" not in pyproject:
    pyproject["project"] = {}

if "dependencies" not in pyproject["project"]:
    pyproject["project"]["dependencies"] = []

# Update dependencies
pyproject["project"]["dependencies"].extend(requirements)

# Remove duplicates, if any
pyproject["project"]["dependencies"] = list(set(pyproject["project"]["dependencies"]))

# Save the updated pyproject.toml
with open("pyproject.toml", "w", encoding="utf-8") as file:
    toml.dump(pyproject, file)

print("Requirements added to pyproject.toml successfully.")
