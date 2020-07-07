The `Auditioner` class is the main entry point for the Audition module. Users pass its constructor a database connection, information about the model groups to be evaluated, and a specification for a filter to prune the worst-performing models.

Other methods allow users to define more complex selection rules, list selected models, or plot results from the selection process.

::: triage.component.audition.__init__
    rendering:
        show_root_toc_entry: False
        group_by_category: True
        show_category_heading: True
        show_if_no_docstring: True