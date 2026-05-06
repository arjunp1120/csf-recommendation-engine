### 1. Historical Execution & Flow Data (Identifying Willingness)

*   **Historical RFQ (Request for Quote) Metadata:** 
    *   *Data points:* Desk ID / Legal Entity Identifier (LEI), historical response rate (e.g., "Desk A responds to 80% of crude block RFQs"), quote latency (in milliseconds), and quote competitiveness (variance from the lit market midpoint). 
*   **Block Trade Prints (CME ClearPort / ICE Block Data):** 
    *   *Data points:* Timestamp of block prints, volume size, price relative to the Central Limit Order Book (CLOB) at the time of execution. You use this to identify the time of day block liquidity is highest and correlate it with known actors active during those windows.
*   **Post-Trade Market Impact Metrics:** 
    *   *Data points:* Price movement 1, 5, and 60 minutes *after* a counterparty takes a block. If the market aggressively drops immediately after you sell to a specific LEI, the data suggests they are a "toxic" counterparty who immediately dumps the risk onto the lit screen. You exclude them from the pool.

### 2. Microstructure & Order Book Data (Identifying Latent Capacity)

*   **Level 2 / Level 3 (MBO - Market By Order) Data:** 
    *   *Data points:* Resting liquidity deep in the order book. An entity consistently resting 50-100 lots on the bid several ticks down is a prime candidate to be contacted for a 500-lot off-screen block, as the data proves they have the risk appetite.
*   **Order Cancellation Rates (Message-to-Trade Ratios):**
    *   *Data points:* High cancellation rates usually indicate high-frequency trading (HFT) firms. For a 500-lot block, you generally want to filter *out* extreme HFT data profiles in favor of slower, well-capitalized bank desks or fundamental macro funds.

### 3. Fundamental & Physical Market Data (Identifying the "Natural" Buyer)

*   **Refinery Turnaround and Utilization Data:** 
    *   *Data points:* Schedules of when major refineries are coming out of maintenance. A refinery ramping up operations has a fundamental data flag indicating a need to buy crude (or hedge future crude purchases like a Jun 26 contract).
*   **Shipping and Logistics Data (AIS Tracking):** 
    *   *Data points:* Tanker charter rates and routing data. Entities charting Very Large Crude Carriers (VLCCs) for future delivery dates have a verifiable, data-backed need to hedge those cargoes. 
*   **Commitments of Traders (COT) Reports:** 
    *   *Data points:* Aggregate open interest broken down by "Producer/Merchant/Processor/User" vs. "Managed Money." While this doesn't give you individual names, tracking the week-over-week change in this data tells you which *category* of market participant currently has the appetite to absorb a 500-lot short.

### 4. Counterparty Credit & Clearing Data (The Risk Filters)
*   **FCM (Futures Commission Merchant) Margin Limits API:**
    *   *Data points:* Real-time Initial Margin (IM) surplus/deficit data for the counterparty's clearing account. Do they possess the sheer free cash to cover the IM for 500 lots of Crude?
*   **Credit Default Swap (CDS) Spreads:** 
    *   *Data points:* Daily CDS spreads of the candidate's parent bank or trading house. A sudden spike in an entity's CDS spread is an automatic data trigger to remove them from your candidate pool due to institutional risk.
*   **KYC/AML Jurisdiction Databases:**
    *   *Data points:* Legal entity registration codes, LEI status (active/lapsed), and sanctions list API cross-referencing.