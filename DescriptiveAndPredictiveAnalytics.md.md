---


---

<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> numpy <span class="token keyword">as</span> np
<span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd
</code></pre>
<pre class=" language-python"><code class="prism  language-python">listing_df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'C:\\DataScience\\00_AnalysisProjects\\AirbnbAnalysis\\Data\\London_listings.csv'</span><span class="token punctuation">)</span>
listing_df<span class="token punctuation">.</span>shape   <span class="token comment"># (77096, 96)</span>
</code></pre>
<pre><code>C:\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2785: DtypeWarning: Columns (61,62,87) have mixed types. Specify dtype option on import or set low_memory=False.
  interactivity=interactivity, compiler=compiler, result=result)





(77096, 96)
</code></pre>
<pre class=" language-python"><code class="prism  language-python">calendar_df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'C:\\DataScience\\00_AnalysisProjects\\AirbnbAnalysis\\Data\\London_calendar.csv'</span><span class="token punctuation">)</span>
calendar_df<span class="token punctuation">.</span>shape 
</code></pre>
<pre><code>(28139675, 4)
</code></pre>
<pre class=" language-python"><code class="prism  language-python">calendar <span class="token operator">=</span> calendar_df
calendar<span class="token punctuation">[</span><span class="token string">"price"</span><span class="token punctuation">]</span> <span class="token operator">=</span> calendar<span class="token punctuation">[</span><span class="token string">"price"</span><span class="token punctuation">]</span><span class="token punctuation">.</span><span class="token builtin">apply</span><span class="token punctuation">(</span><span class="token keyword">lambda</span> x<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">.</span>replace<span class="token punctuation">(</span><span class="token string">"$"</span><span class="token punctuation">,</span> <span class="token string">""</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
calendar<span class="token punctuation">[</span><span class="token string">"price"</span><span class="token punctuation">]</span> <span class="token operator">=</span> pd<span class="token punctuation">.</span>to_numeric<span class="token punctuation">(</span>calendar<span class="token punctuation">[</span><span class="token string">"price"</span><span class="token punctuation">]</span> <span class="token punctuation">,</span> errors<span class="token operator">=</span><span class="token string">"coerce"</span><span class="token punctuation">)</span>
df1  <span class="token operator">=</span> calendar<span class="token punctuation">.</span>groupby<span class="token punctuation">(</span><span class="token string">"date"</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token string">"price"</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span>
df1<span class="token punctuation">[</span><span class="token string">"mean"</span><span class="token punctuation">]</span>  <span class="token operator">=</span> calendar<span class="token punctuation">.</span>groupby<span class="token punctuation">(</span><span class="token string">"date"</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token string">"price"</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span>
df1<span class="token punctuation">.</span>columns <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">"Total"</span><span class="token punctuation">,</span> <span class="token string">"Average"</span><span class="token punctuation">]</span>
df1<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<div>
</div>
<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>Total</th>
      <th>Average</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-12-07</th>
      <td>686632.0</td>
      <td>145.411266</td>
    </tr>
    <tr>
      <th>2018-12-08</th>
      <td>1191258.0</td>
      <td>131.094751</td>
    </tr>
    <tr>
      <th>2018-12-09</th>
      <td>2021083.0</td>
      <td>117.457023</td>
    </tr>
    <tr>
      <th>2018-12-10</th>
      <td>2326273.0</td>
      <td>116.968675</td>
    </tr>
    <tr>
      <th>2018-12-11</th>
      <td>2400023.0</td>
      <td>116.687233</td>
    </tr>
  </tbody>
</table>

<pre class=" language-python"><code class="prism  language-python">df2 <span class="token operator">=</span> calendar<span class="token punctuation">.</span>set_index<span class="token punctuation">(</span><span class="token string">"date"</span><span class="token punctuation">)</span>
df2<span class="token punctuation">.</span>index <span class="token operator">=</span> pd<span class="token punctuation">.</span>to_datetime<span class="token punctuation">(</span>df2<span class="token punctuation">.</span>index<span class="token punctuation">)</span>
df2 <span class="token operator">=</span>  df2<span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token string">"price"</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">.</span>resample<span class="token punctuation">(</span><span class="token string">"M"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span>
df2<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<div>
</div>
<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>price</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-12-31</th>
      <td>122.479159</td>
    </tr>
    <tr>
      <th>2019-01-31</th>
      <td>114.149572</td>
    </tr>
    <tr>
      <th>2019-02-28</th>
      <td>113.226533</td>
    </tr>
    <tr>
      <th>2019-03-31</th>
      <td>120.287040</td>
    </tr>
    <tr>
      <th>2019-04-30</th>
      <td>126.576012</td>
    </tr>
  </tbody>
</table>

<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> plotly <span class="token keyword">as</span> py
<span class="token keyword">from</span> plotly<span class="token punctuation">.</span>offline <span class="token keyword">import</span> iplot<span class="token punctuation">,</span> plot<span class="token punctuation">,</span> init_notebook_mode<span class="token punctuation">,</span> download_plotlyjs
<span class="token keyword">import</span> plotly<span class="token punctuation">.</span>graph_objs <span class="token keyword">as</span> go
init_notebook_mode<span class="token punctuation">(</span>connected<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
<span class="token keyword">import</span> plotly<span class="token punctuation">.</span>offline <span class="token keyword">as</span> offline
</code></pre>

<pre class=" language-python"><code class="prism  language-python">trace1 <span class="token operator">=</span> go<span class="token punctuation">.</span>Scatter<span class="token punctuation">(</span>
    x <span class="token operator">=</span> df1<span class="token punctuation">.</span>index<span class="token punctuation">,</span>
    y <span class="token operator">=</span> df1<span class="token punctuation">[</span><span class="token string">"Total"</span><span class="token punctuation">]</span>
<span class="token punctuation">)</span>
data <span class="token operator">=</span> <span class="token punctuation">[</span>trace1<span class="token punctuation">]</span>
layout <span class="token operator">=</span> go<span class="token punctuation">.</span>Layout<span class="token punctuation">(</span>
    title <span class="token operator">=</span> <span class="token string">"Price by each time"</span><span class="token punctuation">,</span>
    xaxis  <span class="token operator">=</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>title <span class="token operator">=</span> <span class="token string">"Time"</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    yaxis <span class="token operator">=</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>title <span class="token operator">=</span> <span class="token string">"Total ($)"</span><span class="token punctuation">)</span>
<span class="token punctuation">)</span>
trace2 <span class="token operator">=</span> go<span class="token punctuation">.</span>Scatter<span class="token punctuation">(</span>
    x <span class="token operator">=</span> df1<span class="token punctuation">.</span>index<span class="token punctuation">,</span>
    y <span class="token operator">=</span> df1<span class="token punctuation">[</span><span class="token string">"Average"</span><span class="token punctuation">]</span>
<span class="token punctuation">)</span>

data2 <span class="token operator">=</span> <span class="token punctuation">[</span>trace2<span class="token punctuation">]</span>
layout2 <span class="token operator">=</span> go<span class="token punctuation">.</span>Layout<span class="token punctuation">(</span>
    title <span class="token operator">=</span> <span class="token string">"Price by each time"</span><span class="token punctuation">,</span>
    xaxis  <span class="token operator">=</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>title <span class="token operator">=</span> <span class="token string">"Time"</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    yaxis <span class="token operator">=</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>title <span class="token operator">=</span> <span class="token string">"Mean ($)"</span><span class="token punctuation">)</span>
<span class="token punctuation">)</span>
fig <span class="token operator">=</span> go<span class="token punctuation">.</span>Figure<span class="token punctuation">(</span>data <span class="token operator">=</span> data<span class="token punctuation">,</span> layout <span class="token operator">=</span> layout<span class="token punctuation">)</span>
fig2 <span class="token operator">=</span> go<span class="token punctuation">.</span>Figure<span class="token punctuation">(</span>data <span class="token operator">=</span> data2<span class="token punctuation">,</span> layout <span class="token operator">=</span> layout2<span class="token punctuation">)</span>
offline<span class="token punctuation">.</span>iplot<span class="token punctuation">(</span>fig<span class="token punctuation">)</span>
</code></pre>
<div id="a92802ea-0b78-4398-9bed-185e00b8822a" class="plotly-graph-div"></div>
<pre class=" language-python"><code class="prism  language-python">offline<span class="token punctuation">.</span>iplot<span class="token punctuation">(</span>fig2<span class="token punctuation">)</span>
</code></pre>
<div id="ff9facd5-377d-4dda-ba47-06f756d24e19" class="plotly-graph-div"></div>
<pre class=" language-python"><code class="prism  language-python">trace3 <span class="token operator">=</span> go<span class="token punctuation">.</span>Scatter<span class="token punctuation">(</span>
    x <span class="token operator">=</span> df2<span class="token punctuation">.</span>index<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
    y <span class="token operator">=</span> df2<span class="token punctuation">.</span>price<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span>
<span class="token punctuation">)</span>
layout3 <span class="token operator">=</span> go<span class="token punctuation">.</span>Layout<span class="token punctuation">(</span>
    title <span class="token operator">=</span> <span class="token string">"Average price by month"</span><span class="token punctuation">,</span>
    xaxis <span class="token operator">=</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>title <span class="token operator">=</span> <span class="token string">"time"</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    yaxis <span class="token operator">=</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>title <span class="token operator">=</span> <span class="token string">"Price"</span><span class="token punctuation">)</span>
<span class="token punctuation">)</span>
data3 <span class="token operator">=</span> <span class="token punctuation">[</span>trace3<span class="token punctuation">]</span>
fig3 <span class="token operator">=</span> go<span class="token punctuation">.</span>Figure<span class="token punctuation">(</span>data<span class="token operator">=</span> data3<span class="token punctuation">,</span> layout<span class="token operator">=</span> layout3<span class="token punctuation">)</span>
offline<span class="token punctuation">.</span>iplot<span class="token punctuation">(</span>fig3<span class="token punctuation">)</span>
</code></pre>
<div id="81e913a9-34bf-4b96-90df-b4f56f082bef" class="plotly-graph-div"></div>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> statsmodels<span class="token punctuation">.</span>tsa<span class="token punctuation">.</span>seasonal <span class="token keyword">import</span> seasonal_decompose
</code></pre>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">draw_interactive_graph</span><span class="token punctuation">(</span>mode<span class="token punctuation">)</span><span class="token punctuation">:</span>
    df1<span class="token punctuation">.</span>index <span class="token operator">=</span> pd<span class="token punctuation">.</span>to_datetime<span class="token punctuation">(</span>df1<span class="token punctuation">.</span>index<span class="token punctuation">)</span>
    decomposition <span class="token operator">=</span> seasonal_decompose<span class="token punctuation">(</span>df1<span class="token punctuation">[</span><span class="token punctuation">[</span>mode<span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
    trace4_1 <span class="token operator">=</span> go<span class="token punctuation">.</span>Scatter<span class="token punctuation">(</span>
        x <span class="token operator">=</span> decomposition<span class="token punctuation">.</span>observed<span class="token punctuation">.</span>index<span class="token punctuation">,</span> 
        y <span class="token operator">=</span> decomposition<span class="token punctuation">.</span>observed<span class="token punctuation">[</span>mode<span class="token punctuation">]</span><span class="token punctuation">,</span>
        name <span class="token operator">=</span> <span class="token string">"Observed"</span>
    <span class="token punctuation">)</span>
    trace4_2 <span class="token operator">=</span> go<span class="token punctuation">.</span>Scatter<span class="token punctuation">(</span>
        x <span class="token operator">=</span> decomposition<span class="token punctuation">.</span>trend<span class="token punctuation">.</span>index<span class="token punctuation">,</span>
        y <span class="token operator">=</span> decomposition<span class="token punctuation">.</span>trend<span class="token punctuation">[</span>mode<span class="token punctuation">]</span><span class="token punctuation">,</span>
        name <span class="token operator">=</span> <span class="token string">"Trend"</span>
    <span class="token punctuation">)</span>
    trace4_3 <span class="token operator">=</span> go<span class="token punctuation">.</span>Scatter<span class="token punctuation">(</span>
        x <span class="token operator">=</span> decomposition<span class="token punctuation">.</span>seasonal<span class="token punctuation">.</span>index<span class="token punctuation">,</span>
        y <span class="token operator">=</span> decomposition<span class="token punctuation">.</span>seasonal<span class="token punctuation">[</span>mode<span class="token punctuation">]</span><span class="token punctuation">,</span>
        name <span class="token operator">=</span> <span class="token string">"Seasonal"</span>
    <span class="token punctuation">)</span>
    trace4_4 <span class="token operator">=</span> go<span class="token punctuation">.</span>Scatter<span class="token punctuation">(</span>
        x <span class="token operator">=</span> decomposition<span class="token punctuation">.</span>resid<span class="token punctuation">.</span>index<span class="token punctuation">,</span>
        y <span class="token operator">=</span> decomposition<span class="token punctuation">.</span>resid<span class="token punctuation">[</span>mode<span class="token punctuation">]</span><span class="token punctuation">,</span>
        name <span class="token operator">=</span> <span class="token string">"Resid"</span>
    <span class="token punctuation">)</span>

    fig <span class="token operator">=</span> py<span class="token punctuation">.</span>tools<span class="token punctuation">.</span>make_subplots<span class="token punctuation">(</span>rows<span class="token operator">=</span><span class="token number">4</span><span class="token punctuation">,</span> cols<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">,</span> subplot_titles<span class="token operator">=</span><span class="token punctuation">(</span><span class="token string">'Observed'</span><span class="token punctuation">,</span> <span class="token string">'Trend'</span><span class="token punctuation">,</span>
                                                              <span class="token string">'Seasonal'</span><span class="token punctuation">,</span> <span class="token string">'Residiual'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    <span class="token comment"># append trace into fig</span>
    fig<span class="token punctuation">.</span>append_trace<span class="token punctuation">(</span>trace4_1<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>
    fig<span class="token punctuation">.</span>append_trace<span class="token punctuation">(</span>trace4_2<span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>
    fig<span class="token punctuation">.</span>append_trace<span class="token punctuation">(</span>trace4_3<span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>
    fig<span class="token punctuation">.</span>append_trace<span class="token punctuation">(</span>trace4_4<span class="token punctuation">,</span> <span class="token number">4</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>

    fig<span class="token punctuation">[</span><span class="token string">'layout'</span><span class="token punctuation">]</span><span class="token punctuation">.</span>update<span class="token punctuation">(</span> title<span class="token operator">=</span><span class="token string">'Descompose with TimeSeri'</span><span class="token punctuation">)</span>
    offline<span class="token punctuation">.</span>iplot<span class="token punctuation">(</span>fig<span class="token punctuation">)</span>
</code></pre>
<pre class=" language-python"><code class="prism  language-python">draw_interactive_graph<span class="token punctuation">(</span><span class="token string">"Average"</span><span class="token punctuation">)</span>
</code></pre>
<pre><code>This is the format of your plot grid:
[ (1,1) x1,y1 ]
[ (2,1) x2,y2 ]
[ (3,1) x3,y3 ]
[ (4,1) x4,y4 ]
</code></pre>
<div id="af8cb5d4-9a24-4b1e-ab14-06e228e96e83" class="plotly-graph-div"></div>
<pre class=" language-python"><code class="prism  language-python">draw_interactive_graph<span class="token punctuation">(</span><span class="token string">"Total"</span><span class="token punctuation">)</span>
</code></pre>
<pre><code>This is the format of your plot grid:
[ (1,1) x1,y1 ]
[ (2,1) x2,y2 ]
[ (3,1) x3,y3 ]
[ (4,1) x4,y4 ]
</code></pre>
<div id="9073f490-1d20-4aeb-8c32-94f130dc2b73" class="plotly-graph-div"></div>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">loc_city</span><span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">if</span> <span class="token string">","</span> <span class="token operator">not</span> <span class="token keyword">in</span> <span class="token builtin">str</span><span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> x
    <span class="token keyword">if</span> <span class="token string">"live"</span> <span class="token keyword">in</span> <span class="token builtin">str</span><span class="token punctuation">(</span>x<span class="token punctuation">)</span> <span class="token operator">or</span> <span class="token string">"Next door to"</span> <span class="token keyword">in</span> <span class="token builtin">str</span><span class="token punctuation">(</span>x<span class="token punctuation">)</span> <span class="token operator">or</span> <span class="token string">"live"</span> <span class="token keyword">in</span> <span class="token builtin">str</span><span class="token punctuation">(</span>x<span class="token punctuation">)</span> <span class="token operator">or</span> <span class="token string">"having"</span> <span class="token keyword">in</span> <span class="token builtin">str</span><span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> <span class="token string">"USA"</span>
    <span class="token keyword">return</span> <span class="token builtin">str</span><span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">.</span>split<span class="token punctuation">(</span><span class="token string">","</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span>
a <span class="token operator">=</span> listing_df<span class="token punctuation">[</span><span class="token string">"host_location"</span><span class="token punctuation">]</span><span class="token punctuation">.</span><span class="token builtin">apply</span><span class="token punctuation">(</span><span class="token keyword">lambda</span> x<span class="token punctuation">:</span> loc_city<span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<pre class=" language-python"><code class="prism  language-python">df_listing <span class="token operator">=</span> listing_df<span class="token punctuation">[</span>listing_df<span class="token punctuation">.</span>applymap<span class="token punctuation">(</span>np<span class="token punctuation">.</span>isreal<span class="token punctuation">)</span><span class="token punctuation">]</span>
df_listing<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span>how <span class="token operator">=</span> <span class="token string">"all"</span><span class="token punctuation">,</span> axis <span class="token operator">=</span> <span class="token number">1</span><span class="token punctuation">,</span> inplace <span class="token operator">=</span> <span class="token boolean">True</span><span class="token punctuation">)</span>
df_listing<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token punctuation">)</span>
df_listing<span class="token punctuation">[</span><span class="token string">"City"</span><span class="token punctuation">]</span>  <span class="token operator">=</span> a
df_listing<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span>
</code></pre>
<div>
</div>
<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>id</th>
      <th>scrape_id</th>
      <th>host_id</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>...</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>license</th>
      <th>calculated_host_listings_count</th>
      <th>reviews_per_month</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9554</td>
      <td>20181207034825</td>
      <td>31655</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>51.587767</td>
      <td>-0.105666</td>
      <td>2</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>4</td>
      <td>1.65</td>
      <td>London</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 31 columns</p>

<pre class=" language-python"><code class="prism  language-python">df_seattle <span class="token operator">=</span> df_listing<span class="token punctuation">[</span>df_listing<span class="token punctuation">[</span><span class="token string">"City"</span><span class="token punctuation">]</span> <span class="token operator">==</span> <span class="token string">"Seattle"</span><span class="token punctuation">]</span>
df_seattle<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<div>
</div>
<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>id</th>
      <th>scrape_id</th>
      <th>host_id</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>...</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>license</th>
      <th>calculated_host_listings_count</th>
      <th>reviews_per_month</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9804</th>
      <td>6550729</td>
      <td>20181207034825</td>
      <td>4271330</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>51.538811</td>
      <td>-0.045712</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0.07</td>
      <td>Seattle</td>
    </tr>
    <tr>
      <th>24803</th>
      <td>14279479</td>
      <td>20181207034825</td>
      <td>20786100</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>51.563350</td>
      <td>-0.123351</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>0.11</td>
      <td>Seattle</td>
    </tr>
    <tr>
      <th>25497</th>
      <td>14644267</td>
      <td>20181207034825</td>
      <td>17597728</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>51.485301</td>
      <td>-0.073172</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>1.17</td>
      <td>Seattle</td>
    </tr>
    <tr>
      <th>37104</th>
      <td>19351472</td>
      <td>20181207034825</td>
      <td>20786100</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>51.561813</td>
      <td>-0.123615</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>0.60</td>
      <td>Seattle</td>
    </tr>
    <tr>
      <th>69921</th>
      <td>29114534</td>
      <td>20181207034825</td>
      <td>17597728</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>51.485174</td>
      <td>-0.073388</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>Seattle</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>

<pre class=" language-python"><code class="prism  language-python">calendar_clean <span class="token operator">=</span> calendar<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span><span class="token punctuation">)</span>
calendar_clean<span class="token punctuation">.</span>set_index<span class="token punctuation">(</span><span class="token string">"date"</span><span class="token punctuation">,</span> inplace <span class="token operator">=</span> <span class="token boolean">True</span><span class="token punctuation">)</span>
calendar_clean<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<div>
</div>
<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>listing_id</th>
      <th>available</th>
      <th>price</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-11-25</th>
      <td>9554</td>
      <td>t</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>2019-11-24</th>
      <td>9554</td>
      <td>t</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>2019-11-23</th>
      <td>9554</td>
      <td>t</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>2019-11-22</th>
      <td>9554</td>
      <td>t</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>2019-11-21</th>
      <td>9554</td>
      <td>t</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>

<pre class=" language-python"><code class="prism  language-python">calendar_clean<span class="token punctuation">.</span>index <span class="token operator">=</span> pd<span class="token punctuation">.</span>to_datetime<span class="token punctuation">(</span>calendar_clean<span class="token punctuation">.</span>index<span class="token punctuation">)</span>
number_hire_room <span class="token operator">=</span> calendar_clean<span class="token punctuation">.</span>resample<span class="token punctuation">(</span><span class="token string">"M"</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token string">"price"</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">.</span>count<span class="token punctuation">(</span><span class="token punctuation">)</span>
total_price_each_month  <span class="token operator">=</span> calendar_clean<span class="token punctuation">.</span>resample<span class="token punctuation">(</span><span class="token string">"M"</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token string">"price"</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<pre class=" language-python"><code class="prism  language-python">trace5 <span class="token operator">=</span> go<span class="token punctuation">.</span>Scatter<span class="token punctuation">(</span>
    x <span class="token operator">=</span> number_hire_room<span class="token punctuation">.</span>index<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
    y <span class="token operator">=</span> number_hire_room<span class="token punctuation">.</span>price<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span>
<span class="token punctuation">)</span>
data5 <span class="token operator">=</span> <span class="token punctuation">[</span>trace5<span class="token punctuation">]</span>
layout5 <span class="token operator">=</span> go<span class="token punctuation">.</span>Layout<span class="token punctuation">(</span>
    title <span class="token operator">=</span> <span class="token string">"Number of Hire Room by Month in Seattle"</span><span class="token punctuation">,</span>
    xaxis <span class="token operator">=</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>title <span class="token operator">=</span> <span class="token string">"Month"</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    yaxis <span class="token operator">=</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>title <span class="token operator">=</span> <span class="token string">"Number hirde"</span><span class="token punctuation">)</span>
<span class="token punctuation">)</span>
fig5  <span class="token operator">=</span> go<span class="token punctuation">.</span>Figure<span class="token punctuation">(</span>data <span class="token operator">=</span> data5<span class="token punctuation">,</span> layout <span class="token operator">=</span> layout5<span class="token punctuation">)</span>
</code></pre>
<pre class=" language-python"><code class="prism  language-python">trace6 <span class="token operator">=</span> go<span class="token punctuation">.</span>Scatter<span class="token punctuation">(</span>
    x <span class="token operator">=</span> number_hire_room<span class="token punctuation">.</span>index<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
    y <span class="token operator">=</span> number_hire_room<span class="token punctuation">.</span>price<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token operator">/</span>number_hire_room<span class="token punctuation">.</span>price<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span>
<span class="token punctuation">)</span>
data6 <span class="token operator">=</span> <span class="token punctuation">[</span>trace6<span class="token punctuation">]</span>
layout6 <span class="token operator">=</span> go<span class="token punctuation">.</span>Layout<span class="token punctuation">(</span>
    title <span class="token operator">=</span> <span class="token string">"the ratio of the number of rooms compare with the first month"</span><span class="token punctuation">,</span>
    xaxis <span class="token operator">=</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>title <span class="token operator">=</span> <span class="token string">"Month"</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    yaxis <span class="token operator">=</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>title <span class="token operator">=</span> <span class="token string">"Ratio"</span><span class="token punctuation">)</span>
<span class="token punctuation">)</span>
fig6 <span class="token operator">=</span> go<span class="token punctuation">.</span>Figure<span class="token punctuation">(</span>data <span class="token operator">=</span> data6<span class="token punctuation">,</span> layout <span class="token operator">=</span> layout6<span class="token punctuation">)</span>
</code></pre>
<pre class=" language-python"><code class="prism  language-python">offline<span class="token punctuation">.</span>iplot<span class="token punctuation">(</span>fig5<span class="token punctuation">)</span>
</code></pre>
<div id="38b2df27-f0e7-4f3a-9490-3c54b270aa22" class="plotly-graph-div"></div>
<pre class=" language-python"><code class="prism  language-python">offline<span class="token punctuation">.</span>iplot<span class="token punctuation">(</span>fig6<span class="token punctuation">)</span>
</code></pre>
<div id="6d28e93b-26cb-4aef-9fe6-5fafc2286a16" class="plotly-graph-div"></div>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> scipy <span class="token keyword">import</span> stats
</code></pre>
<pre class=" language-python"><code class="prism  language-python">a <span class="token operator">=</span> calendar_clean<span class="token punctuation">.</span>index<span class="token punctuation">.</span>month
<span class="token comment"># calendar_clean["Month"] = a</span>
calendar_clean <span class="token operator">=</span> calendar_clean<span class="token punctuation">.</span>assign<span class="token punctuation">(</span>Month <span class="token operator">=</span> a<span class="token punctuation">)</span>
calendar_clean<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<div>
</div>
<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>listing_id</th>
      <th>available</th>
      <th>price</th>
      <th>Month</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-11-25</th>
      <td>9554</td>
      <td>t</td>
      <td>35.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2019-11-24</th>
      <td>9554</td>
      <td>t</td>
      <td>35.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2019-11-23</th>
      <td>9554</td>
      <td>t</td>
      <td>39.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2019-11-22</th>
      <td>9554</td>
      <td>t</td>
      <td>39.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2019-11-21</th>
      <td>9554</td>
      <td>t</td>
      <td>35.0</td>
      <td>11</td>
    </tr>
  </tbody>
</table>

<pre class=" language-python"><code class="prism  language-python">

result <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
<span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span><span class="token number">13</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    result<span class="token punctuation">.</span>append<span class="token punctuation">(</span>np<span class="token punctuation">.</span>array<span class="token punctuation">(</span><span class="token punctuation">[</span>calendar_clean<span class="token punctuation">[</span>calendar_clean<span class="token punctuation">[</span><span class="token string">"Month"</span><span class="token punctuation">]</span> <span class="token operator">==</span> i<span class="token punctuation">]</span><span class="token punctuation">.</span>price<span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">)</span>


</code></pre>
<pre class=" language-python"><code class="prism  language-python">data_score <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
<span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token number">11</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    score <span class="token operator">=</span> stats<span class="token punctuation">.</span>ttest_rel<span class="token punctuation">(</span>result<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token number">64911</span><span class="token punctuation">]</span><span class="token punctuation">,</span>result<span class="token punctuation">[</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token number">64911</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
    data_score<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">(</span>score<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> score<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<pre class=" language-python"><code class="prism  language-python">score_board <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span>data <span class="token operator">=</span> data_score<span class="token punctuation">,</span> columns <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">"Test Statistic"</span><span class="token punctuation">,</span> <span class="token string">"P_value"</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
score_board<span class="token punctuation">[</span><span class="token string">"Month"</span><span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">12</span><span class="token punctuation">)</span>
score_board<span class="token punctuation">.</span>set_index<span class="token punctuation">(</span><span class="token string">"Month"</span><span class="token punctuation">,</span> inplace <span class="token operator">=</span> <span class="token boolean">True</span><span class="token punctuation">)</span>
score_board
</code></pre>
<div>
</div>
<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>Test Statistic</th>
      <th>P_value</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-14.907433</td>
      <td>3.571048e-50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-15.748268</td>
      <td>8.957391e-56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-14.036928</td>
      <td>1.077138e-44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-10.806497</td>
      <td>3.382590e-27</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-11.480491</td>
      <td>1.769421e-30</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-6.952102</td>
      <td>3.632667e-12</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-2.532751</td>
      <td>1.131948e-02</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.698523</td>
      <td>4.848527e-01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-5.631242</td>
      <td>1.796535e-08</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-6.849708</td>
      <td>7.465732e-12</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-7.256005</td>
      <td>4.031294e-13</td>
    </tr>
  </tbody>
</table>

<pre class=" language-python"><code class="prism  language-python">offline<span class="token punctuation">.</span>iplot<span class="token punctuation">(</span>fig3<span class="token punctuation">)</span>
</code></pre>
<div id="2cb7dd33-e270-4b11-9f83-8db1215c2fd8" class="plotly-graph-div"></div>

