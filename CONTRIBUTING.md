Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

### Report Bugs

Report bugs at <https://github.com/dssg/triage/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in
    troubleshooting.
-   Detailed steps to reproduce the bug.

### Write Documentation

Triage could always use more documentation, whether as part of the
official Triage docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/dssg/triage/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to
    implement.

Get Started!
------------

Ready to contribute? Here's how to set up triage for local development.

1.  Fork the triage repo on GitHub.
2.  Clone your fork locally:

        $ git clone git@github.com:your_name_here/triage.git

3.  Install your local copy. You can control the environment any way you'd like (virtualenv, pyenv, etc), but the directions below are for virtualenvwrapper.
    Regardless of your Python environment, we do recommend installing `triage` as a module in development mode, via `python setup.py develop` or `pip install -e .`

        $ mkvirtualenv triage
        $ cd triage/
        $ python setup.py develop

4.  Create a branch for local development:

        $ git checkout -b name-of-your-bugfix-or-feature

5.  Make your changes.

    Triage has several code standards that are worth mentioning here. The write-ups are too long to include in the body of this list, so click the links to view them below on this page.

    - [Small Changes](#small-changes)
    - [Testing](#testing)
    - [Code Style](#code-style)
    - [Validation](#validation)
    - [Experiment Versioning](#experiment-versioning)
    - [Documentation](#documentation)

6.  When you're done making changes, check that your changes pass flake8
    and the tests, including testing other Python versions with tox:

        $ flake8 triage tests
        $ python setup.py test or py.test
        $ tox

    To get flake8 and tox, just pip install them into your virtualenv.

7.  Commit your changes and push your branch to GitHub. If you've made multiple commits, squash them into one. Each pull request should be one commit (unless there are multiple logical places to split up the code, for instance a big reformatting commit followed by a smaller commit that actually changes behavior). To make the pull request easier to create, the last commit on the branch should be formatted like the following.

    ```
    One-line summary [Resolves #issueno]

    Optional longer text description

    - optional enumerated change to file #1 and related files
    - optional enumerated change to file #2 and related files
    ```

    The top line will be prepopulated by Github as the title of the pull request. The [Resolves #issueno] will automatically close the referenced issue when the commit is merged into master. The rest of the text will be prepopulated by Github as the content of the pull request. Ideally, you should put any useful text you want to show up in the pull request in the commit message in this way, because the commit message is available to read in the commit history long after the pull request is inconvenient to dig up. Whether or not you feel the commit is best described by a text writeup, enumerated changes, or a combination of both, the One-line summary is very useful no matter what.

    To achieve this format, it's not recommended to use `git commit -m "message"`. Instead use

        $ git commit

    And [write the commit message in your favorite text editor](https://help.github.com/articles/associating-text-editors-with-git/)

8.  Submit a pull request through the GitHub website.  If you are working directly with the DSaPP team, *assign somebody*! Pull requests should not linger be out for a long time. And tell them that you assigned them.



Small Changes
-------------
Pull Requests take a lot of time to read. Finding defects is not the only reason to have other developers review code, but it is useful to give them a fighting chance to find defects without taking a ton of time to review. Generally, the smaller the pull request, the better. The ideal pull request is the minimal piece of code that adds value. A lot has been written on the subject; one good write-up is [The Anatomy of A Perfect Pull Request](https://medium.com/@hugooodias/the-anatomy-of-a-perfect-pull-request-567382bb6067). 


Testing
-------
Triage has an extensive test suite that is run with every commit by travis-ci. Any pull request will be rejected if it breaks the travis-ci build. To help prevent yourself from encountering annoying test failures at pull request time, we strongly recommend utilizing 'Test-Driven Development' (TDD) to introduce tests along with code.

Many of the tests require a Postgres server, so you need the Postgres server executable installed on the system you're developing on. You don't need to create any databases, or even have the server running; the tests that need it take care of starting a server, creating databases, and tearing it all down when they're done.

For a general workflow applying TDD to Triage, consider this example:

Let's say I have decided to change the behavior of the LabelGenerator class in [src/triage/component/architect/label_generators.py](src/triage/component/architect/label_generators.py). The unit test, following the standard within `triage`, will be in [src/tests/architect_tests/test_label_generators.py](src/tests/architect_tests/test_label_generators.py).

### 1. Run the test first
Before making any code changes, I run this specific test with:

    $ py.test src/tests/architect_tests/test_label_generators.py 

Running it before writing any code helps as a sanity check to make sure my environment is working correctly and the test hasn't been broken.

Obviously you can skip this step if you are adding a completely new, untested class or function.

### 2. Modify the test so it fails

Provided it works, I now want to look at changing the test (or adding a new one, if this is new code entirely) to reflect the new behavior I expect the class to have. Unit tests are supposed to define expectations of the behavior of the code under test, and my changes to the expected behavior should be reflected in the test. In many cases, this will involve modifying the input or output (or both) of an existing unit test. In other cases, if I want to add a new behavior but leave the existing behavior as an option, I would want to add a new test that covers the new behavior option only.

Running this test again should cause it to fail, because the test has the new behavior and the code it's testing has the old behavior. This step is important because I want to be sure that the test cares enough about the code's behavior to notice that it doesn't match up with expectations.

Adding or updating comments to reflect the test's behavior will be helpful to future developers in the same file as you.

### 3. Change the code to make the test pass

Now I enter the original code file and modify it so that the code reflects my updated expecations, and I run the test again to make sure it passes.

### 4. Repeat with all known affected pieces of code

Some changes only affect one class or function, and that's great! However, sometimes I know up-front that you need to change more than one file. For instance, if I add a required argument to the constructor of the LabelGenerator class, it will need to be added to anything that instantiates a LabelGenerator, such as the Experiment.  The Experiment tests should actually fail at this point, so I can fix that code and run the Experiment test to make sure it works.

### 5. Run the whole suite

There may be side effects of code changes that I don't know about at first. The test suite takes a while to run, so I don't want to run it after every code change I make, but it is a good idea to run it as a sanity check when I'm done with my changes and am ready to pull request.

Code Style
----------
Triage follows the PEP8 code style standards. This is validated with flake8 every commit by travis-ci. Any pull request will be rejected if it introduces code that breaks the travis-ci build. To help prevent yourself from encountering annoying style failures at pull request time, you can run flake8 regularly during code development. You can use Triage's flake8 settings by running it through tox, like so:

    $ pip install flake8 tox
    $ tox -e flake8 src/triage/component/architect/label_generators.py

You may also consider installing flake8 integration in your code editor of choice to be informed of style violations even more quickly.


Validation
----------

The larger Triage entrypoints (e.g. Experiment, Audition) are largely driven by serializable configuration, and implement validation routines so users can validate that configuration without running computationally-intensive code. It is important that the Triage development process keeps these validation routines up to date so users can continue to rely on them.

- **Experiment** - If a code changes adds to or changes Experiment configuration values, the [experiment validation](src/triage/experiments/validate.py) class should be updated. A best effort should be made to raise any issues with the experiment definition that can be caught before running the full experiment. This can involve checking values against whitelists, inspecting given database tables, or even running some version of given queries through an EXPLAIN command.
- **Audition** - If a code changes or adds to Audition configuration values, the [validate method of the AuditionRunner class](src/triage/component/audition/__init__.py) should be updated.


Experiment Versioning
---------------------
If the change breaks old experiment definitions, the experiment config version should be updated in both the [example experiment config](example_experiment_config.yaml) and [experiment module base](src/triage/experiments/__init__.py)

Documentation
-------------

- All classes/methods should have docstrings describing their purpose
- If the code adds or changes experiment configuration values, the [example\_experiment\_config.yaml](example_experiment_config.yaml) should be updated with the new or changed experiment configuration, and instructions for using it.
- If the code adds to or changes the behavior of the Experiment, update:
    - [Experiment Algorithm Page](docs/sources/experiments/algorithm.md)
    - [DirtyDuck Tutorial](https://www.github.com/dssg/dirtyduck)
