(require 'package)

(when (<= emacs-major-version 27)
  (package-initialize) ;; Not needed in Emacs 27
                                        ; Disable loading package again after init.el
  )


(unless (package-installed-p 'use-package)
  (package-refresh-contents)
  (package-install 'use-package))

(eval-when-compile
  (require 'use-package))

(use-package htmlize
  :defer t
  )

(require 'ox-publish)
(setq org-publish-project-alist
      '(

        ("dirtyduck-notes"
         :base-directory "~/projects/dsapp/dirtyduck/org/"
         :base-extension "org"
         :exclude "[[:digit:]][[:digit:]]_.*\.org"
         :publishing-directory "~/projects/dsapp/dirtyduck/docs/"
         :recursive t
         :publishing-function org-html-publish-to-html
         :headline-levels 4       ; Just the default for this project.
         :auto-preamble t
         :sitemap-title "Dirtyduck"
         )

        ("dirtyduck-notes-md"
         :base-directory "~/projects/dsapp/dirtyduck/org/"
         :base-extension "org"
         :exclude "[[:digit:]][[:digit:]]_.*\.org"
         :publishing-directory "~/projects/dsapp/dirtyduck/docs/"
         :recursive t
         :publishing-function org-gfm-export-to-markdown
         :headline-levels 4       ; Just the default for this project.
         :auto-preamble t
         :sitemap-title "Dirtyduck"
         )

        ("dirtyduck-static"
         :base-directory "~/projects/dsapp/dirtyduck/org/"
         :base-extension "css\\|js\\|png\\|jpg\\|gif\\|pdf\\|mp3\\|ogg\\|swf\\|sql\\|svg\\|yaml"
         :publishing-directory "~/projects/dsapp/dirtyduck/docs/"
         :recursive t
         :publishing-function org-publish-attachment
         )


        ("dirtyduck" :components ("dirtyduck-static" "dirtyduck-notes"  "dirtyduck-notes-md"))

        ))
