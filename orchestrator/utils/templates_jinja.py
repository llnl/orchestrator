import os
from jinja2 import Environment, FileSystemLoader
from typing import Optional


def render_template_to_file(
    template_path: str,
    output_dir: str,
    replacements: dict[str, str],
    output_file_name: Optional[str] = None,
):
    """
    Renders a Jinja2 template with replacements and writes to an output file.

    :param template_path: full path to the template file
    :param output_dir: directory to save the output file
    :param replacements: dictionary of variable names and their replacement
        values
    :param output_file_name: optional output file name. Defaults to template
        file name
    :returns: name of the generated file (not including the full path)
    """
    if output_file_name is None:
        output_file_name = os.path.basename(template_path)
    output_path = os.path.join(output_dir, output_file_name)
    file_count = 1

    # Avoid overwriting existing files
    name, ext = os.path.splitext(output_file_name)
    while os.path.isfile(output_path):
        output_file_name = f'{name}_{file_count}{ext}'
        output_path = os.path.join(output_dir, output_file_name)
        file_count += 1

    # Set up Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(template_path) or '.'))
    template = env.get_template(os.path.basename(template_path))
    rendered_content = template.render(**replacements)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'wt') as out_file:
        out_file.write(rendered_content)

    return output_file_name
