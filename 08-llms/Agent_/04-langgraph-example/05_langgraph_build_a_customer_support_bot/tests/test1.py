from langchain_core.tools import tool

# Example 1: Simple tool that adds two numbers
@tool
def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        The sum of a and b
    """
    return a + b

# Example 2: Tool that gets current weather for a location
@tool
def get_weather(city: str) -> str:
    """Gets the current weather for a given city.
    
    Args:
        city: The city to get weather for
        
    Returns:
        String describing the current weather
    """
    # In a real implementation, you would call a weather API here
    return f"The weather in {city} is sunny and 72°F"

# Example 3: Tool with more complex functionality
@tool
def search_products(query: str, max_results: int = 5) -> list:
    """Searches for products matching a query.
    
    Args:
        query: Search term for products
        max_results: Maximum number of results to return
        
    Returns:
        List of product names matching the query
    """
    # Mock implementation - in real usage you might call an API or database
    return [f"Product {i} ({query})" for i in range(1, max_results+1)]

# You can then use these tools in your LangChain agent
tools = [add_numbers, get_weather, search_products]

# Example usage:
print(add_numbers.invoke({"a": 5, "b": 3}))  # Output: 8
print(get_weather.invoke({"city": "New York"}))  # Output: The weather in New York is sunny and 72°F