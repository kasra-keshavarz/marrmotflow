Templating Module
=================

The templating module provides functions for generating MARRMOT model configuration files using Jinja2 templates.

.. automodule:: marrmotflow.templating
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

render_models
~~~~~~~~~~~~~

.. autofunction:: marrmotflow.templating.render_models

   Generate MARRMOT model configuration files from templates.

   :param model_files: Sequence of model file names to generate
   :type model_files: Sequence[str]
   :param template_jinja_path: Path to Jinja2 template file
   :type template_jinja_path: PathLike, optional
   :return: Dictionary mapping file names to generated content
   :rtype: Dict[str, str]

   **Example:**

   .. code-block:: python

      from marrmotflow.templating import render_models

      # Generate model files using default template
      model_files = ["model_7.m", "model_37.m"]
      rendered = render_models(model_files)

      # Access generated content
      for filename, content in rendered.items():
          print(f"Generated {filename}")
          print(content[:100] + "...")

   **Custom Template:**

   .. code-block:: python

      # Use custom template
      custom_template = "my_template.m.jinja"
      rendered = render_models(
          model_files=["custom_model.m"],
          template_jinja_path=custom_template
      )

raise_helper
~~~~~~~~~~~~

.. autofunction:: marrmotflow.templating.raise_helper

   Jinja2 helper function for raising exceptions within templates.

   :param msg: Error message to raise
   :type msg: str
   :raises Exception: Always raises an Exception with the provided message

   This function is registered as a global in the Jinja2 environment and can be
   used within templates for error handling:

   .. code-block:: jinja2

      {% if not parameters %}
      {{ raise("No parameters defined for model") }}
      {% endif %}

Constants
---------

.. autodata:: marrmotflow.templating.TEMPLATE_MODEL
   :annotation: = "marrmot_models.m.jinja"

   Default template file name for MARRMOT models.

Template Environment
--------------------

The module sets up a Jinja2 environment with the following configuration:

.. autodata:: marrmotflow.templating.environment

   Global Jinja2 environment configured for MARRMOT template processing.

   **Configuration:**
   
   - **Loader**: PackageLoader("marrmotflow", "templates")
   - **trim_blocks**: True
   - **lstrip_blocks**: True  
   - **line_comment_prefix**: '##'

   **Global Variables:**

   - **raise**: Helper function for raising exceptions in templates

Template Structure
------------------

Default Template Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

The default template expects the following variables to be available:

.. list-table:: Template Variables
   :header-rows: 1
   :widths: 20 20 60

   * - Variable
     - Type
     - Description
   * - model_name
     - str
     - Human-readable name of the model
   * - model_number
     - int
     - MARRMOT model structure number
   * - catchment_id
     - str
     - Unique identifier for the catchment
   * - catchment_name
     - str
     - Human-readable catchment name
   * - parameters
     - List[Dict]
     - List of model parameters with name, value, description
   * - forcing_data
     - Dict[str, str]
     - Mapping of forcing variable names to data identifiers
   * - timestamp
     - str
     - Generation timestamp
   * - workflow_name
     - str
     - Name of the workflow generating the template

Template Syntax
~~~~~~~~~~~~~~~

The templates use Jinja2 syntax with the following customizations:

**Line Comments:**

.. code-block:: jinja2

   ## This is a comment that will be removed from output
   % This is MATLAB code that will be preserved

**Block Control:**

.. code-block:: jinja2

   {% for param in parameters %}
   {{ param.name }} = {{ param.value }}; % {{ param.description }}
   {% endfor %}

**Conditional Logic:**

.. code-block:: jinja2

   {% if model_number == 7 %}
   % HBV-96 specific configuration
   {% elif model_number == 37 %}
   % GR4J specific configuration
   {% endif %}

**Error Handling:**

.. code-block:: jinja2

   {% if not parameters %}
   {{ raise("No parameters defined") }}
   {% endif %}

Examples
--------

Basic Template Rendering
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marrmotflow.templating import render_models

   # Render multiple model files
   model_files = ["hbv_model.m", "gr4j_model.m"]
   results = render_models(model_files)

   # Save to files
   for filename, content in results.items():
       with open(f"generated_{filename}", 'w') as f:
           f.write(content)

Custom Template Usage
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create custom template file: my_template.m.jinja
   template_content = '''
   % Custom MARRMOT Model: {{ model_name }}
   % Generated: {{ timestamp }}
   
   function output = {{ model_name|lower|replace('-', '_') }}_model()
       % Model parameters
       {% for param in parameters %}
       {{ param.name }} = {{ param.value }}; % {{ param.description }}
       {% endfor %}
       
       % Model implementation here
       output = struct();
   end
   '''

   # Use custom template
   results = render_models(
       ["custom_model.m"],
       template_jinja_path="my_template.m.jinja"
   )

Template Context Creation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example of template context that would be passed to render_models
   template_context = {
       "model_name": "HBV-96",
       "model_number": 7,
       "catchment_id": "basin_001",
       "catchment_name": "Example Basin",
       "parameters": [
           {
               "name": "TT",
               "value": 0.0,
               "description": "Temperature threshold for snow"
           },
           {
               "name": "C0", 
               "value": 3.0,
               "description": "Degree-day factor"
           }
       ],
       "forcing_data": {
           "precipitation": "precip_data",
           "temperature": "temp_data"
       },
       "timestamp": "2024-07-16 12:00:00",
       "workflow_name": "ExampleWorkflow"
   }

Error Handling
--------------

Template Errors
~~~~~~~~~~~~~~~

The templating system can raise several types of errors:

.. py:exception:: jinja2.TemplateNotFound

   Raised when the specified template file cannot be found:

   .. code-block:: python

      try:
          render_models(["model.m"], "nonexistent_template.jinja")
      except jinja2.TemplateNotFound as e:
          print(f"Template not found: {e}")

.. py:exception:: jinja2.TemplateSyntaxError

   Raised when the template contains syntax errors:

   .. code-block:: python

      # Invalid template syntax will raise TemplateSyntaxError
      # Example: unclosed block, invalid variable name, etc.

.. py:exception:: Exception

   Raised by the raise_helper function within templates:

   .. code-block:: python

      # Template with: {{ raise("Custom error message") }}
      # Will raise: Exception: Custom error message

Best Practices
--------------

1. **Validate template syntax** before deployment
2. **Use descriptive variable names** in templates
3. **Include error checking** with raise_helper
4. **Document template variables** and their expected types
5. **Test templates** with various input contexts
6. **Version control templates** along with code
7. **Use meaningful file names** for generated models

Type Definitions
----------------

.. autodata:: marrmotflow.templating.PathLike
   :annotation:

   Type alias for file paths, supporting both string and os.PathLike objects.
