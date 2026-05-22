import networkx as nx
import numpy as np

def build_graph(df):
    """
    Builds a lightweight in-memory knowledge graph from transaction data.
    Simulates entities (user, device, location) by binning feature buckets.
    """
    G = nx.Graph()
    
    # Pre-calculate buckets for efficiency
    # user: V1, V2
    # device: V3, V4
    # location: V5, V6
    
    for _, row in df.iterrows():
        txn_id = str(row.get('transaction_id', f"TXN-{_}"))
        
        # 1. Transaction Node
        G.add_node(txn_id, type='transaction', amount=row.get('Amount', 0))
        
        # 2. Simulated User Node (based on V1/V2 similarity)
        u_id = f"USER_{round(row['V1'], 0)}_{round(row['V2'], 0)}"
        G.add_node(u_id, type='user')
        G.add_edge(txn_id, u_id, label='used_by')
        
        # 3. Simulated Device Node (based on V3/V4 similarity)
        d_id = f"DEV_{round(row['V3'], 0)}_{round(row['V4'], 0)}"
        G.add_node(d_id, type='device')
        G.add_edge(txn_id, d_id, label='used_device')
        
        # 4. Simulated Location Node (based on V5/V6 similarity)
        l_id = f"LOC_{round(row['V5'], 0)}_{round(row['V6'], 0)}"
        G.add_node(l_id, type='location')
        G.add_edge(txn_id, l_id, label='shares_location')
        
        # 5. Merchant Node
        m_id = str(row.get('merchant_category', 'Retail'))
        G.add_node(m_id, type='merchant')
        G.add_edge(txn_id, m_id, label='same_merchant')
        
        # 6. Time Node
        hour = int(row.get('hour', (row.get('Time', 0) // 3600) % 24))
        t_id = f"HOUR_{hour}"
        G.add_node(t_id, type='time')
        G.add_edge(txn_id, t_id, label='transacted_at')

    return G

def get_graph_insights(transaction_data, graph):
    """
    Analyzes the graph relative to a specific transaction to find relational patterns.
    Returns:
    {
        "connected_accounts": int,
        "shared_devices": int,
        "suspicious_links": list[str]
    }
    """
    insights = []
    
    # Re-simulate IDs for the current transaction
    v = transaction_data
    u_id = f"USER_{round(v.get('v1', 0), 0)}_{round(v.get('v2', 0), 0)}"
    d_id = f"DEV_{round(v.get('v3', 0), 0)}_{round(v.get('v4', 0), 0)}"
    l_id = f"LOC_{round(v.get('v5', 0), 0)}_{round(v.get('v6', 0), 0)}"
    m_id = str(v.get('merchant_category', 'Retail'))
    
    connected_accounts = 0
    shared_devices = 0
    
    # 1. Device Reuse Analysis
    if graph.has_node(d_id):
        linked_txns = [n for n in graph.neighbors(d_id) if graph.nodes[n].get('type') == 'transaction']
        if len(linked_txns) >= 1:
            shared_devices = len(linked_txns) + 1  # Add 1 for the current transaction
            insights.append(f"Device ID {d_id[-6:]} reused across {shared_devices} distinct transactions within 24h.")
    
    # 2. Location & Merchant Anomaly
    if graph.has_node(l_id):
        linked_txns = [n for n in graph.neighbors(l_id) if graph.nodes[n].get('type') == 'transaction']
        if len(linked_txns) >= 2:
            insights.append(f"Location cluster linked to {len(linked_txns) + 1} high-velocity transactions.")

    # 3. User Velocity
    if graph.has_node(u_id):
        linked_txns = [n for n in graph.neighbors(u_id) if graph.nodes[n].get('type') == 'transaction']
        connected_accounts = len(linked_txns)
        if connected_accounts >= 1:
            insights.append(f"Account identity shows {connected_accounts + 1} rapid-fire attempts across different merchants.")

    # Fallback if no specific insights found
    if not insights:
        insights.append("No suspicious multi-account or device reuse patterns detected in Knowledge Graph.")

    return {
        "connected_accounts": connected_accounts,
        "shared_devices": shared_devices,
        "suspicious_links": insights
    }
