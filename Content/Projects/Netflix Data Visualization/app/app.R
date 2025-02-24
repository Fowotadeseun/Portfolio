library(shiny)
library(ggplot2)
library(dplyr)
library(plotly)
library(shinythemes)
library(readr)
library(htmlwidgets)

netflix_data <- read_csv("netflix1.csv") %>%
  mutate(date_added = as.Date(date_added, format = "%m/%d/%Y")) %>%
  mutate(year_added = as.integer(format(date_added, "%Y"))) %>%
  filter(!is.na(year_added))

ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("Netflix Data Analysis: Trends and Insights (2008-2021)"),
  p("Welcome to the Netflix Data Analysis Dashboard! This interactive platform allows you to explore trends in Netflix's content library from 2008 to 2021. Use the filters to analyze content types, geographic distribution, and rating patterns over time."),
  sidebarLayout(
    sidebarPanel(
      selectInput("type", "Select Content Type:", choices = c("All", "Movie", "TV Show")),
      selectInput("country", "Select Country:", choices = c("All", unique(netflix_data$country))),
      sliderInput("yearRange", "Select Year Range:", min = 2008, max = 2021, value = c(2008, 2021), step = 1),
      helpText("Use the controls to filter the data based on content type, country, and year range.")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("📊 Content Types", plotlyOutput("contentTypePlot", height = "600px")),
        tabPanel("🌎 Geographic Diversity", plotlyOutput("geoDiversityPlot", height = "600px")),
        tabPanel("⭐ Content Ratings", plotlyOutput("contentRatingPlot", height = "600px"))
      )
    )
  )
)

server <- function(input, output) {
  filtered_data <- reactive({
    data <- netflix_data
    if (input$type != "All") {
      data <- data %>% filter(type == input$type)
    }
    if (input$country != "All") {
      data <- data %>% filter(country == input$country)
    }
    data <- data %>% filter(year_added >= input$yearRange[1] & year_added <= input$yearRange[2])
    return(data)
  })
  
  output$contentTypePlot <- renderPlotly({
    data <- filtered_data() %>%
      group_by(year_added, type) %>%
      summarize(count = n(), .groups = 'drop')
    
    p <- ggplot(data, aes(x = year_added, y = count, color = type, group = type)) +
      geom_line(size = 1.5) +
      geom_point(size = 3) +
      labs(title = "Trends in Content Types on Netflix", x = "Year", y = "Count") +
      theme_minimal(base_size = 14)
    
    ggplotly(p) %>%
      layout(height = 600)
  })
  
  output$geoDiversityPlot <- renderPlotly({
    data <- filtered_data() %>%
      group_by(year_added, country) %>%
      summarize(count = n(), .groups = 'drop')
    
    p <- ggplot(data, aes(x = year_added, y = count, fill = country)) +
      geom_bar(stat = "identity", show.legend = FALSE) +
      labs(title = "Geographic Diversity of Content on Netflix", x = "Year", y = "Count") +
      theme_minimal(base_size = 14)
    
    ggplotly(p) %>%
      layout(height = 600)
  })
  
  output$contentRatingPlot <- renderPlotly({
    data <- filtered_data() %>%
      group_by(rating) %>%
      summarize(count = n(), .groups = 'drop')
    
    p <- ggplot(data, aes(x = rating, y = count, fill = rating)) +
      geom_bar(stat = "identity", show.legend = FALSE) +
      labs(title = "Trends in Content Ratings on Netflix", x = "Rating", y = "Count") +
      theme_minimal(base_size = 14)
    
    ggplotly(p) %>%
      layout(height = 600)
  })
}

shinyApp(ui = ui, server = server)
