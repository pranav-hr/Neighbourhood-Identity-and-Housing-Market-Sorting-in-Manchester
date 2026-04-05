# ============================================================
# Text Embedding Pipeline — Manchester Housing Project
# ============================================================
py_install("sentence-transformers", pip = TRUE)
install.packages("tidytext")
library(tidytext)
library(tidyverse)
library(httr)
install.packages("jsonlite")
library(jsonlite)
install.packages("reticulate")
library(reticulate)

# ── 0. Load your data ────────────────────────────────────────
df <- read_csv("final_df.csv")

msoa_codes <- df %>% distinct(msoa21cd)


# ── 1. Get MSOA → readable name crosswalk (ONS) ──────────────
# House of Commons Library MSOA Names (publicly available)
msoa_names_url <- "https://houseofcommonslibrary.github.io/msoanames/MSOA-Names-2.2.csv"

msoa_names <- read_csv(msoa_names_url) %>%
  select(msoa21cd, msoa21hclnm)   # e.g. "Didsbury West", "Moss Side"

# Join readable names onto your MSOA codes
msoa_lookup <- msoa_codes %>%
  left_join(msoa_names, by = "msoa21cd")

print(head(msoa_lookup, 10))

# ── 2. Fetch Wikipedia summaries ─────────────────────────────
get_wiki_text <- function(place_name) {
  if (is.na(place_name)) return(NA_character_)
  
  base_url <- "https://en.wikipedia.org/w/api.php"
  
  # Split compound names like "Openshaw & Gorton North" into parts
  parts <- str_split(place_name, "\\s*&\\s*")[[1]] %>% str_trim()
  
  # Build query list: compound name first, then each individual part,
  # then each part stripped of directional suffix
  queries <- c(
    place_name,
    parts,
    str_remove(parts, "\\s+(North|South|East|West|Central|Upper|Lower)$") %>% str_trim()
  ) %>% unique()
  
  for (q in queries) {
    resp <- tryCatch(
      GET(
        url = base_url,
        query = list(
          action      = "query",
          prop        = "extracts",
          explaintext = "true",
          redirects   = "1",
          format      = "json",
          titles      = q
        ),
        add_headers(`User-Agent` = "ManchesterHousingProject/1.0 (academic@example.com)"),
        timeout(15)
      ),
      error = function(e) NULL
    )
    
    if (is.null(resp)) next
    if (status_code(resp) != 200) next
    
    raw_text <- rawToChar(resp$content)
    if (startsWith(trimws(raw_text), "<")) next
    
    content <- tryCatch(fromJSON(raw_text), error = function(e) NULL)
    if (is.null(content)) next
    
    pages <- content$query$pages
    page  <- pages[[1]]
    
    if (!is.null(page$extract) && nchar(page$extract) > 50) {
      return(page$extract)
    }
    
    Sys.sleep(0.3)
  }
  
  return(NA_character_)
}

# Fetch for all 59 MSOAs
wiki_results <- msoa_lookup %>%
  mutate(wiki_text = map_chr(msoa21hclnm, get_wiki_text))

# Report coverage
n_found <- sum(!is.na(wiki_results$wiki_text))
n_missing <- sum(is.na(wiki_results$wiki_text))

# Show which ones failed
if (n_missing > 0) {
  cat("\nMissing Wikipedia text for:\n")
  wiki_results %>% filter(is.na(wiki_text)) %>% print()
}

# Fill NAs with neutral placeholder
wiki_results <- wiki_results %>%
  mutate(wiki_text = replace_na(wiki_text, "Residential neighbourhood in Manchester, Greater Manchester, England."))

#Save
write_csv(wiki_results, "msoa_wiki_texts.csv")

# ============================================================
# Fix incorrect / missing Wikipedia articles for problem MSOAs
# ============================================================


# ── Helper: fetch full Wikipedia article by EXACT title ──────
fetch_wiki_by_title <- function(title) {
  resp <- tryCatch(
    GET(
      url = "https://en.wikipedia.org/w/api.php",
      query = list(
        action      = "query",
        prop        = "extracts",
        explaintext = "true",
        redirects   = "1",
        format      = "json",
        titles      = title
      ),
      add_headers(`User-Agent` = "ManchesterHousingProject/1.0 (academic@example.com)"),
      timeout(15)
    ),
    error = function(e) NULL
  )
  
  if (is.null(resp) || status_code(resp) != 200) return(NA_character_)
  
  raw_text <- rawToChar(resp$content)
  if (startsWith(trimws(raw_text), "<")) return(NA_character_)
  
  content <- tryCatch(fromJSON(raw_text), error = function(e) NULL)
  if (is.null(content)) return(NA_character_)
  
  page <- content$query$pages[[1]]
  if (!is.null(page$extract) && nchar(page$extract) > 50) page$extract else NA_character_
}

# ── Manual override table ─────────────────────────────────────
# Maps each problem MSOA code to the correct Wikipedia article title
# For compound MSOAs, we concatenate both articles

overrides <- tribble(
  ~msoa21cd,    ~correct_titles,
  
  # Wrong article fetched
  "E02006984",  list("Bradford Moor", "New Islington, Manchester"),   # got Bradford city
  "E02006912",  list("Ancoats", "Piccadilly, Manchester"),             # got Piccadilly London
  "E02001091",  list("Baguley", "Brooklands, Manchester"),             # got Brooklands circuit
  "E02006914",  list("Oxford Road Corridor", "Whitworth Street"),      # got Croatian university
  "E02006916",  list("Hulme", "St George's, Manchester"),              # got Saint George martyr
  "E02001072",  list("Whalley Range, Manchester"),                     # got disambiguation
  "E02001066",  list("Victoria Park, Manchester", "Longsight"),        # got disambiguation
  "E02001047",  list("Charlestown, Manchester"),                       # got disambiguation
  "E02001073",  list("Chorlton-cum-Hardy"),                            # got disambiguation
  "E02001077",  list("Chorlton-cum-Hardy"),                            # got disambiguation
  "E02006986",  list("Levenshulme"),                                   # got cardinal direction
  "E02001064",  list("Belle Vue, Manchester", "West Gorton"),          # got disambiguation
  "E02001081",  list("Chorlton-cum-Hardy", "Chorlton Ees"),            # got Gore Brook stream
  "E02001051",  list("Moston, Manchester"),                            # got disambiguation
  "E02001096",  list("Peel Hall, Wythenshawe"),                        # got disambiguation
  
  # Placeholder / too thin - try better search terms
  "E02001087",  list("Didsbury"),                                      # placeholder
  "E02001059",  list("Beswick, Manchester", "Openshaw"),               # placeholder
)

# ── Fetch correct text for each override ─────────────────────
override_texts <- overrides %>%
  mutate(
    wiki_text = map_chr(correct_titles, function(titles) {
      # Fetch each title and concatenate
      texts <- map_chr(titles, function(t) {
        cat("  Fetching:", t, "\n")
        txt <- fetch_wiki_by_title(t)
        Sys.sleep(0.3)
        if (is.na(txt)) "" else txt
      })
      combined <- paste(texts[nchar(texts) > 0], collapse = "\n\n")
      if (nchar(combined) > 50) combined else NA_character_
    })
  ) %>%
  select(msoa21cd, wiki_text)

# Report
override_texts %>%
  mutate(status = if_else(is.na(wiki_text), "STILL MISSING", "Fixed")) %>%
  print()


# ── Apply overrides to wiki_results ──────────────────────────
# wiki_results should already be loaded from your main script
# If re-running standalone, load it:
# wiki_results <- read_csv("msoa_wiki_texts.csv")

wiki_results_fixed <- wiki_results %>%
  left_join(override_texts, by = "msoa21cd", suffix = c("", "_new")) %>%
  mutate(
    wiki_text = if_else(!is.na(wiki_text_new), wiki_text_new, wiki_text)
  ) %>%
  select(-wiki_text_new)



# Check final coverage
n_placeholder <- sum(wiki_results_fixed$wiki_text == 
                       "Residential neighbourhood in Manchester, Greater Manchester, England.")

wiki_results_fixed %>%
  mutate(chars = nchar(wiki_text)) %>%
  arrange(chars) %>%
  select(msoa21hclnm, chars) %>%
  print(n = 59)


# Replace wiki_results in your environment for the rest of the pipeline
wiki_results <- wiki_results_fixed


# Clean Wikipedia formatting noise before encoding
wiki_results <- wiki_results %>%
  mutate(wiki_text = wiki_text %>%
           # Remove section headers like == History == or === Transport ===
           str_remove_all("={2,}[^=]+=={2,}") %>%
           # Remove excess newlines
           str_replace_all("\\n{2,}", " ") %>%
           str_replace_all("\\n", " ") %>%
           # Remove citation artifacts like [1] [2]
           str_remove_all("\\[\\d+\\]") %>%
           # Remove "may refer to:" disambiguation remnants
           str_remove_all("may refer to:.*") %>%
           # Collapse multiple spaces
           str_replace_all("\\s{2,}", " ") %>%
           str_trim()
  )

# Overwrite saved CSV
write_csv(wiki_results, "msoa_wiki_textsnew2.csv")

# Replace wiki_results in your environment for the rest of the pipeline
wiki_results <- wiki_results_fixed

wiki_results <- read_csv("msoa_wiki_textsnew2.csv") %>%
  mutate(wiki_text = wiki_text %>%
           # Remove == section headers == in all their variations
           str_remove_all("={2,}\\s*[^=\n]+\\s*={2,}") %>%
           # Remove leftover newlines
           str_replace_all("\\n+", " ") %>%
           # Remove [1] [2] citation markers
           str_remove_all("\\[\\d+\\]") %>%
           # Collapse multiple spaces
           str_replace_all("\\s{2,}", " ") %>%
           str_trim()
  )

# ── 3. TF-IDF vectors ────────────────────────────────────────

# Tokenise each neighbourhood's text into words
tfidf_matrix <- wiki_results %>%
  filter(wiki_text != "Residential neighbourhood in Manchester, Greater Manchester, England.") %>%
  unnest_tokens(word, wiki_text) %>%
  # Remove stop words (the, a, and, etc.)
  anti_join(stop_words, by = "word") %>%
  # Remove pure numbers
  filter(!str_detect(word, "^\\d+$")) %>%
  # Count word frequency per MSOA
  count(msoa21cd, word) %>%
  # Compute TF-IDF
  bind_tf_idf(word, msoa21cd, n) %>%
  # Keep only TF-IDF score
  select(msoa21cd, word, tf_idf) %>%
  # Pivot to wide matrix: rows = MSOAs, columns = words
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0)

# Extract just the numeric matrix for PCA
msoa_ids    <- tfidf_matrix$msoa21cd
tfidf_nums  <- tfidf_matrix %>% select(-msoa21cd) %>% as.matrix()

# Rows = MSOAs, Columns = unique words

# ── 4. PCA to compress ───────────────────────────────────────
tfidf_pca <- prcomp(tfidf_nums, center = TRUE, scale. = FALSE)
# Note: scale. = FALSE for TF-IDF as variables are already on same scale

# Variance explained
cumvar <- cumsum(summary(tfidf_pca)$importance[2, 1:50])
print(round(cumvar, 3))

# Keep first 25 components for comparability with text embeddings
tfidf_emb_df <- as_tibble(tfidf_pca$x[, 1:25]) %>%
  rename_with(~ paste0("tfidf_emb", 1:25)) %>%
  mutate(msoa21cd = msoa_ids)

# ── 5. Merge into final_df ───────────────────────────────────
df <- df %>%
  left_join(tfidf_emb_df, by = "msoa21cd")

df <- df %>% filter(!is.na(tfidf_emb1))

write_csv(df, "final_df_withtfidf_embeddings.csv")
