"""Few-shot prompt registry for text-to-SQL.

Injects verified SQL examples into the system prompt based on the database
schema. These examples teach the model patterns that are specific to this
schema (e.g., how to join InvoiceLine to Genre, how to compute revenue,
how to rank customers) — pushing accuracy from 90% to 95%+.

The registry is organized by pattern type so we can inject only the most
relevant examples (keeping the prompt short) or all examples for maximum
coverage.
"""

from typing import Optional


# ── Verified SQL examples for the Chinook schema ──────────────────────────────
# Each example has: question, SQL, pattern_type, and an optional explanation.
# These are verified against the actual Chinook database and produce correct
# results. They cover the most common query patterns the model will encounter.

FEW_SHOT_EXAMPLES = [
    {
        "pattern": "aggregation_join",
        "question": "What are the top 5 best-selling genres by total sales?",
        "sql": (
            "SELECT g.Name AS Genre, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS TotalSales\n"
            "FROM InvoiceLine il\n"
            "JOIN Track t ON il.TrackId = t.TrackId\n"
            "JOIN Genre g ON t.GenreId = g.GenreId\n"
            "GROUP BY g.GenreId, g.Name\n"
            "ORDER BY TotalSales DESC\n"
            "LIMIT 5;"
        ),
        "explanation": "Join InvoiceLine -> Track -> Genre to aggregate sales by genre. Use ROUND for money.",
    },
    {
        "pattern": "filter_join",
        "question": "List all albums by the artist 'AC/DC'.",
        "sql": (
            "SELECT a.Title AS AlbumTitle\n"
            "FROM Album a\n"
            "JOIN Artist ar ON a.ArtistId = ar.ArtistId\n"
            "WHERE ar.Name = 'AC/DC'\n"
            "ORDER BY a.Title;"
        ),
        "explanation": "Join Album -> Artist and filter by artist name.",
    },
    {
        "pattern": "count_groupby",
        "question": "How many customers does each employee support?",
        "sql": (
            "SELECT e.FirstName || ' ' || e.LastName AS EmployeeName,\n"
            "       COUNT(c.CustomerId) AS CustomerCount\n"
            "FROM Employee e\n"
            "LEFT JOIN Customer c ON e.EmployeeId = c.SupportRepId\n"
            "GROUP BY e.EmployeeId, e.FirstName, e.LastName\n"
            "ORDER BY CustomerCount DESC;"
        ),
        "explanation": "Use LEFT JOIN so employees with 0 customers still appear. Combine FirstName+LastName.",
    },
    {
        "pattern": "date_filter_aggregation",
        "question": "What is the total revenue generated in the year 2021?",
        "sql": (
            "SELECT ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS TotalRevenue\n"
            "FROM Invoice i\n"
            "JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId\n"
            "WHERE strftime('%Y', i.InvoiceDate) = '2021';"
        ),
        "explanation": "Use strftime() for date filtering in SQLite. Revenue = SUM(UnitPrice * Quantity).",
    },
    {
        "pattern": "window_function_ranking",
        "question": "Who are the top 5 customers by total spending, ranked?",
        "sql": (
            "SELECT c.FirstName || ' ' || c.LastName AS CustomerName,\n"
            "       ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS TotalSpending,\n"
            "       RANK() OVER (ORDER BY SUM(il.UnitPrice * il.Quantity) DESC) AS Rank\n"
            "FROM Customer c\n"
            "JOIN Invoice i ON c.CustomerId = i.CustomerId\n"
            "JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId\n"
            "GROUP BY c.CustomerId, c.FirstName, c.LastName\n"
            "ORDER BY Rank\n"
            "LIMIT 5;"
        ),
        "explanation": "Use RANK() window function for ranking. Join Customer -> Invoice -> InvoiceLine for spending.",
    },
    {
        "pattern": "having_clause",
        "question": "Which artists have tracks in more than 3 different genres?",
        "sql": (
            "SELECT ar.Name AS ArtistName, COUNT(DISTINCT t.GenreId) AS GenreCount\n"
            "FROM Artist ar\n"
            "JOIN Album a ON ar.ArtistId = a.ArtistId\n"
            "JOIN Track t ON a.AlbumId = t.AlbumId\n"
            "GROUP BY ar.ArtistId, ar.Name\n"
            "HAVING COUNT(DISTINCT t.GenreId) > 3\n"
            "ORDER BY GenreCount DESC;"
        ),
        "explanation": "Use HAVING for post-aggregation filtering. COUNT(DISTINCT ...) for unique genre count.",
    },
    {
        "pattern": "left_join_each",
        "question": "How many tracks are in each playlist?",
        "sql": (
            "SELECT p.Name AS PlaylistName, COUNT(pt.TrackId) AS TrackCount\n"
            "FROM Playlist p\n"
            "LEFT JOIN PlaylistTrack pt ON p.PlaylistId = pt.PlaylistId\n"
            "GROUP BY p.PlaylistId, p.Name\n"
            "ORDER BY TrackCount DESC;"
        ),
        "explanation": "LEFT JOIN ensures playlists with 0 tracks appear. 'each' implies all rows from the left table.",
    },
    {
        "pattern": "avg_groupby_country",
        "question": "What is the average invoice total for each country?",
        "sql": (
            "SELECT c.Country, ROUND(AVG(i.Total), 2) AS AvgInvoiceTotal\n"
            "FROM Customer c\n"
            "JOIN Invoice i ON c.CustomerId = i.CustomerId\n"
            "GROUP BY c.Country\n"
            "ORDER BY AvgInvoiceTotal DESC;"
        ),
        "explanation": "Join Customer -> Invoice, group by country, use AVG and ROUND for averages.",
    },
    # ── Examples added from groundTruth failure analysis (2026-07-31) ─────────
    # Target the worst clusters: L3/L2 JN (join-path errors), L2/L3 GA (aggregation),
    # L3 FL (filter via correct join path). Verified against Chinook.
    {
        "pattern": "artist_track_count_join",
        "question": "How many tracks does the artist 'A Cor Do Som' have?",
        "sql": (
            "SELECT COUNT(*) AS TrackCount\n"
            "FROM Track t\n"
            "JOIN Album al ON t.AlbumId = al.AlbumId\n"
            "JOIN Artist ar ON al.ArtistId = ar.ArtistId\n"
            "WHERE ar.Name = 'A Cor Do Som';"
        ),
        "explanation": "To count an artist's tracks, join Track -> Album -> Artist and filter on Artist.Name. Do NOT use Track.Composer (it is a free-text field, not the artist) and do NOT guess an ArtistId.",
    },
    {
        "pattern": "artist_track_list_join",
        "question": "List the track names by the artist 'Gilberto Gil'.",
        "sql": (
            "SELECT t.Name AS TrackName\n"
            "FROM Track t\n"
            "JOIN Album al ON t.AlbumId = al.AlbumId\n"
            "JOIN Artist ar ON al.ArtistId = ar.ArtistId\n"
            "WHERE ar.Name = 'Gilberto Gil'\n"
            "ORDER BY t.Name;"
        ),
        "explanation": "Artist's tracks are reached only through Track -> Album -> Artist. Filter on ar.Name, never on a hard-coded ArtistId.",
    },
    {
        "pattern": "revenue_from_invoice_total",
        "question": "What is the total revenue for each customer support representative?",
        "sql": (
            "SELECT e.FirstName || ' ' || e.LastName AS EmployeeName,\n"
            "       ROUND(SUM(i.Total), 2) AS TotalRevenue\n"
            "FROM Employee e\n"
            "JOIN Customer c ON e.EmployeeId = c.SupportRepId\n"
            "JOIN Invoice i ON c.CustomerId = i.CustomerId\n"
            "GROUP BY e.EmployeeId, e.FirstName, e.LastName\n"
            "ORDER BY TotalRevenue DESC;"
        ),
        "explanation": "Revenue per rep: Employee -> Customer (SupportRepId) -> Invoice, summing Invoice.Total. Use Invoice.Total for order-level revenue; only use InvoiceLine.UnitPrice*Quantity when you need line-level detail (e.g. by genre/track).",
    },
    {
        "pattern": "count_distinct_across_join",
        "question": "Which playlists contain tracks from more than 5 different artists?",
        "sql": (
            "SELECT p.Name AS PlaylistName, COUNT(DISTINCT ar.ArtistId) AS ArtistCount\n"
            "FROM Playlist p\n"
            "JOIN PlaylistTrack pt ON p.PlaylistId = pt.PlaylistId\n"
            "JOIN Track t ON pt.TrackId = t.TrackId\n"
            "JOIN Album al ON t.AlbumId = al.AlbumId\n"
            "JOIN Artist ar ON al.ArtistId = ar.ArtistId\n"
            "GROUP BY p.PlaylistId, p.Name\n"
            "HAVING COUNT(DISTINCT ar.ArtistId) > 5\n"
            "ORDER BY ArtistCount DESC;"
        ),
        "explanation": "COUNT(DISTINCT ...) across a multi-join chain (Playlist -> PlaylistTrack -> Track -> Album -> Artist) with HAVING for the threshold. Reach Artist via Album, not directly from Track.",
    },
    {
        "pattern": "avg_track_length_seconds",
        "question": "What is the average length of 'Latin' tracks in seconds?",
        "sql": (
            "SELECT ROUND(AVG(t.Milliseconds) / 1000.0, 1) AS AvgSeconds\n"
            "FROM Track t\n"
            "JOIN Genre g ON t.GenreId = g.GenreId\n"
            "WHERE g.Name = 'Latin';"
        ),
        "explanation": "Track length is stored as Milliseconds; divide by 1000.0 for seconds and ROUND to 1 decimal. Join Track -> Genre to filter by genre name.",
    },
    {
        "pattern": "subquery_above_average",
        "question": "Which customers spent more than the average customer?",
        "sql": (
            "WITH CustomerSpending AS (\n"
            "  SELECT c.CustomerId, c.FirstName || ' ' || c.LastName AS Name,\n"
            "         SUM(i.Total) AS TotalSpent\n"
            "  FROM Customer c\n"
            "  JOIN Invoice i ON c.CustomerId = i.CustomerId\n"
            "  GROUP BY c.CustomerId, c.FirstName, c.LastName\n"
            ")\n"
            "SELECT Name, ROUND(TotalSpent, 2) AS TotalSpent\n"
            "FROM CustomerSpending\n"
            "WHERE TotalSpent > (SELECT AVG(TotalSpent) FROM CustomerSpending)\n"
            "ORDER BY TotalSpent DESC;"
        ),
        "explanation": "Compare each customer to the average with a CTE + scalar subquery on the aggregate. Sum Invoice.Total for order-level spending.",
    },
]


def build_few_shot_block(
    max_examples: int = 5,
    pattern_filter: Optional[list[str]] = None,
) -> str:
    """
    Build a few-shot examples block for injection into the system prompt.

    Args:
        max_examples: Maximum number of examples to include (default 5).
            Keeping this low reduces token usage and latency.
        pattern_filter: Optional list of pattern types to include.
            If None, includes examples across all patterns.

    Returns:
        str: Formatted few-shot examples block ready for prompt injection.
    """
    examples = FEW_SHOT_EXAMPLES

    # Filter by pattern type if specified
    if pattern_filter:
        examples = [e for e in examples if e["pattern"] in pattern_filter]

    # Limit to max_examples
    examples = examples[:max_examples]

    if not examples:
        return ""

    lines = ["## Verified SQL Examples"]
    lines.append("Here are verified SQL queries for this database. Follow these patterns:\n")

    for i, ex in enumerate(examples, 1):
        lines.append(f"### Example {i}: {ex['question']}")
        lines.append(f"```sql\n{ex['sql']}\n```")
        if ex.get("explanation"):
            lines.append(f"Pattern: {ex['pattern']} — {ex['explanation']}")
        lines.append("")

    return "\n".join(lines)


def get_all_patterns() -> list[str]:
    """Return all available pattern types in the registry."""
    return list({e["pattern"] for e in FEW_SHOT_EXAMPLES})


def get_example_count() -> int:
    """Return the total number of examples in the registry."""
    return len(FEW_SHOT_EXAMPLES)
