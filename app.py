import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import time

import CECL_Impact as cecl_ia

#account_data = pd.read_csv("C:/Users/tangji/Desktop/Python/CECL_ia/Jumbo_g.csv")

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, static_file='assets')
application = app.server
app.config['suppress_callback_exceptions']=True

app.layout = html.Div([
                    html.Div([
                                html.Span('C&I Credit Loss Benchmarking Prototype Model (v0.3)',
                                          style={'font-size':40,'color':'#FFE600','margin-left':'2%','font-weight':'bold'}),
                                html.Div(
                                        html.Img(src=app.get_asset_url('EY_Logo_Beam_RGB_White_Yellow.png'),
                                                 style={'height':'80%','vertical-align':'top'}),
                                                 style={'float':'right','height':'100%','margin-right':'2%'})
                                ],style={'height':'70px','line-height':'70px','margin-bottom':'3rem','margin-top':'3rem'}),
        
                    html.Div(
                            children = [
                                    #------------------------------
                                    # left panel with inputs
                                    #-----------------------------
                                    html.Div(        
                                            className='three columns',
                                            children=[
                                                    html.Div(
                                                            children = [
                                                                    html.H5('User Inputs')
                                                                    ],
                                                            style={'border-bottom':'6px solid #FFE600', 'padding-top':20}
                                                            ),
            
                                                    html.Div(
                                                            children = [
                                                                    html.H6('Upload Portfolio Snapsot Data:')
                                                                       ], style={'border-bottom':'4px solid #C4C4CD', 'padding-top':15}
                                                            ),
                                                    
                                                    html.Div(
                                                            children =[
                                                                        dcc.Upload(
                                                                            id='upload_portfolio',
                                                                            children=html.Div([
                                                                                                'Drag and Drop or ',
                                                                                                html.A('Select Files')
                                                                                              ], style={'color': 'white'}),
                                                                            style={
                                                                                    'width': '100%',
                                                                                    'height': '60px',
                                                                                    'lineHeight': '60px',
                                                                                    'borderWidth': '1px',
                                                                                    'borderStyle': 'dashed',
                                                                                    'borderRadius': '5px',
                                                                                    'textAlign': 'center',
                                                                                    'borderColor': 'white',
                                                                                  }
                                                                           ),
                                                                        html.Div(id = 'upload_ind_port') 
                                                                    ],style={'color': 'yellow', 'padding-top': 15}
                                                            ),
                                                            
                                                    html.Div(id='upload_portfolio_store',style={'display':'none'}),
                                                    
                                                    html.Div(
                                                            children=[
                                                                    html.H6('Scenario Input')
                                                                  ],style={'border-bottom':'4px solid #C4C4CD', 'padding-top': 15}
                                                        ),
                                             
                                                    html.Div(
                                                            children=[
                                                                    html.H6('Select Scenario Input Method:', style={'padding-top':15}),
                                                                    dcc.Dropdown(
                                                                                id = 'sc_input_method',
                                                                                options=[
                                                                                        {'label': 'Upload Scenario', 'value': 'upload'},
                                                                                        {'label': 'CCAR 2019 Scenarios', 'value': 'ccar_2019'}
                                                                                ],
                                                                                value='upload'
                                                                                )                  
                                                                  ]
                                                        ),
                                                                                                            
                                                    html.Div(id='sc_input_container', style={'padding-top': 15, 'padding-bottom': 15}),
            
                                                    html.Div(
                                                            children = [
                                                                    html.Button(
                                                                            id = 'btn1',
                                                                            n_clicks=0,
                                                                            children='Upload',
                                                                            style={'background-color':'#FFE600'}
                                                                            )
                                                                    ],
                                                            style={'padding-top':15,'padding-bottom':15}
                                                            ),
                                                    
                                                    html.Div(id='upload_scenario_store',style={'display':'none'}),
                                                    
                                                    html.Div(id='ccar_scenario_store',style={'display':'none'}),
                                        
                                                    html.Div(          
                                                            children = [
                                                                        html.H6('Mean Reversion Approach and R&S Period:')
                                                                       ], style={'border-bottom':'4px solid #C4C4CD', 'padding-top':15}
                                                            ),
                                                    
                                                    html.Div(
                                                            children =[
                                                                    dcc.RadioItems(
                                                                            id = 'reversion',
                                                                            options = [
                                                                                    {'label': 'TTC Matrix', 'value': 'ttc'},
                                                                                    {'label': 'Last PIT Matrix', 'value': 'pit'}
                                                                                    ],
                                                                            value = 'ttc',
                                                                            labelStyle ={'color': 'white'}
                                                                                )
                                                                    ], style={'padding-top':15}
                                                            ),
                                                                
                                                    html.Div(
                                                            children = [
                                                                    html.H6(id = 'rs_slider_output'),
                                                                    dcc.Slider(
                                                                            id = 'rs_len',
                                                                            min = 2,  
                                                                            max = 13,
                                                                            step = 1,
                                                                            value = 9
                                                                            )
                                                                    ],style={'display':'inline-block', 'width': '100%', 'padding-top':15}
                                                            ),
                                                            
                                                    html.Div(
                                                            children = [
                                                                        html.H6('Prepayment Rate Option:'), 
                                                                       ], style={'border-bottom':'4px solid #C4C4CD', 'padding-top':15}
                                                            ),
                                                            
                                                    html.Div(
                                                            children = [
                                                                    dcc.RadioItems(
                                                                            id = 'PP_option',
                                                                            options = [
                                                                                    {'label': 'Prepayment Rate Provided in the Data', 'value': 'pp_data'},
                                                                                    {'label': 'Fixed Prepayment Rate Selected by User', 'value': 'pp_user'}
                                                                                    ],
                                                                            value = 'pp_data',
                                                                            labelStyle ={'color': 'white'}
                                                                                )
                                                                    ], style={'padding-top':15}
                                                            ),
                                                    
                                                    html.Div(
                                                            children = [
                                                                    html.H6(id = 'pp_slider_output'),
                                                                    dcc.Slider(
                                                                            id = 'PP_val',
                                                                            min = 0,  
                                                                            max = 0.2,
                                                                            step = 0.01,
                                                                            value = 0
                                                                            )
                                                                    ],
                                                            ),
                                                                                                           
                                                    html.Div(
                                                            children = [
                                                                        html.H6('LGD Option:')
                                                                       ], style={'border-bottom':'4px solid #C4C4CD', 'padding-top':15}
                                                             ),
                                                    
                                                    html.Div(
                                                            children = [
                                                                        dcc.RadioItems(
                                                                                id = 'LGD_option',
                                                                                options = [
                                                                                        {'label': 'LGD Provided in the Data', 'value': 'lgd_data'},
                                                                                        {'label': 'Frey Jacobs LGD  ', 'value': 'lgd_fj'},
                                                                                        {'label': 'Fixed LGD Selected by User', 'value': 'lgd_user'}
                                                                                        ],
                                                                                value = 'lgd_data',
                                                                                labelStyle ={'color': 'white'}
                                                                                    )
                                                                ], style={'padding-top':15}

                                                            ),
                                                    
                                                    html.Div(
                                                            children = [
                                                                    html.H6(id = 'lgd_slider_output'),
                                                                    dcc.Slider(
                                                                            id = 'LGD_val',
                                                                            min = 0,  
                                                                            max = 1,
                                                                            step = 0.05,
                                                                            value = 0
                                                                            )
                                                                    ],
                                                                    #style={'display':'none'}
                                                                    #style={'display':'inline-block', 'width':370}
                                                            ),
                                                            
                                                    html.Div(
                                                            children = [
                                                                        html.H6('EAD Option:')
                                                                       ], style={'border-bottom':'4px solid #C4C4CD', 'padding-top':15}
                                                             ),
                                                    
                                                    html.Div(
                                                            children = [
                                                                        dcc.RadioItems(
                                                                                id = 'EAD_option',
                                                                                options = [
                                                                                        {'label': 'Amortization On', 'value': 'on'},
                                                                                        {'label': 'Amortization Off', 'value': 'off'}
                                                                                        ],
                                                                                value = 'on',
                                                                                labelStyle ={'color': 'white'}
                                                                                    )
                                                                        ], style={'padding-top':15}
                                                            ),
                                                    
                                                    html.Div(
                                                            children = [
                                                                    html.Button(
                                                                            id = 'btn2',
                                                                            n_clicks=0,
                                                                            children='Run Forecast',
                                                                            style={'background-color':'#FFE600'}
                                                                            ),
                                                                    html.Div(id = 'ecl_frct_ind', style={'color': 'yellow', 'padding-top': 10}) 
                                                                    ],
                                                            style={'padding-top':15,'padding-bottom':15}
                                                            ),
                                                                        
                                                    html.Div(id='ecl_frct_store',style={'display':'none'}),
                                                    
                                                    html.Div(id='pd_frct_store',style={'display':'none'}),
                                                    
                                                    html.Div(id='lgd_frct_store',style={'display':'none'})
                                                    ]),
                                    
                                    #------------------------------
                                    # Middel panel with portfolio and MEV
                                    #------------------------------ 
                                    html.Div(
                                            className = "four columns",
                                            children = [                    
                                                    html.Div(
                                                            children = [
                                                                        html.H5('Portfolio Balance by Industry and Rating ($mm)')
                                                                        ], style = {'padding-top':20, 'border-bottom':'6px solid #FFB46A'}                                                        
                                                            ),
                                                                        
                                                    html.Div(
                                                            children = [
                                                                    dcc.Graph(id='Port_plot')
                                                                    ],
                                                            style = {'padding-top':20} 
                                                            ),
                                                    
                                                    html.Div(
                                                            children = [
                                                                    html.H5('Macroeconomic Variables in the Selected Scenario',style={'padding-top': 15})
                                                                    ],
                                                            style = {'padding-top':20, 'border-bottom':'6px solid #FFB46A'}                                                        
                                                            ),
                                                                        
                                                    html.Div(
                                                            children = [
                                                                    dcc.Graph(id='Scenario_plot')
                                                                    ],
                                                            style = {'padding-top':20} 
                                                            )
                                                    ]
                                            ),
                                    
                                    #------------------------------
                                    # Right panel with forecast output
                                    #------------------------------ 
                                    html.Div(
                                            className = "four columns",
                                            children = [                    
                                                                                                     
                                                     html.Div(
                                                            children = [
                                                                    html.H5('Quarterly PD & LGD Forecast')
                                                                    ],
                                                            style = {'padding-top':20, 'border-bottom':'6px solid #87D3F2'}                                                        
                                                            ),
                                                                        
                                                    html.Div(
                                                            children = [
                                                                    dcc.Graph(id='PD_LGD_plot')
                                                                    ],
                                                            style = {'padding-top':20} 
                                                            ),
                                                    
                                                    html.Div(
                                                            children=[
                                                                    html.H6('PD & LGD Plot Option:', style={'padding-top':15}),
                                                                    dcc.Dropdown(
                                                                                id = 'pd_plot_option',
                                                                                options=[
                                                                                        {'label': 'Portfolio', 'value': 'port'},
                                                                                        {'label': 'Investment Grade', 'value': 'ig'},
                                                                                        {'label': 'Rating 11', 'value': '11'},
                                                                                        {'label': 'Rating 12', 'value': '12'},
                                                                                        {'label': 'Rating 13', 'value': '13'},
                                                                                        {'label': 'Rating 14', 'value': '14'},
                                                                                        {'label': 'Rating 15', 'value': '15'},
                                                                                        {'label': 'Rating 16', 'value': '16'},
                                                                                        {'label': 'Rating 17', 'value': '17'},
                                                                                        {'label': 'Rating 18', 'value': '18'},
                                                                                        {'label': 'Rating 19', 'value': '19'},
                                                                                        {'label': 'Rating 20', 'value': '20'},
                                                                                        {'label': 'Rating 21', 'value': '21'}
                                                                                ],
                                                                                value='port'
                                                                                )
                                                                                    
                                                                  ], style={'padding-top':15}
                                                        ),
                                                    
                                                   html.Div(
                                                            children = [
                                                                    html.H5('Quarterly ECL Forecast ($000s)',style={'padding-top': 15})
                                                                    ],
                                                            style = {'padding-top':20, 'border-bottom':'6px solid #87D3F2'}                                                        
                                                            ),
                                                                        
                                                    html.Div(
                                                            children = [
                                                                    dcc.Graph(id='ECL_plot')
                                                                    ],
                                                            style = {'padding-top':20} 
                                                            )
                                                    
                                                    ]
                                            )
                                    ], style={'margin-left':'2%'}  
                            )
                ]      
        )




#########################################################################################################                        
@app.callback([Output('upload_portfolio_store', 'children'), Output('upload_ind_port', 'children')],
              [Input('upload_portfolio', 'contents')],
              [State('upload_portfolio', 'filename')])
def portfolio1(c, n):
    data = cecl_ia.parse_contents(c, n) 
    
    if data.empty:
        return data.to_json(), 'Upload not successful......'
    else:    
        return data.to_json(), 'Portfolio file [{}] upload successful......'.format(n)

scenario_input = {'upload': html.Div(children = [dcc.Upload(
                                                            id='upload_scenario',
                                                            children=html.Div([
                                                                                'Drag and Drop or ',
                                                                                html.A('Select Files')
                                                                              ], style={'color': 'white'}),
                                                            style={
                                                                    'width': '100%',
                                                                    'height': '60px',
                                                                    'lineHeight': '60px',
                                                                    'borderWidth': '1px',
                                                                    'borderStyle': 'dashed',
                                                                    'borderRadius': '5px',
                                                                    'textAlign': 'center',
                                                                    'borderColor': 'white',
                                                                  }
                                                           ),
                                                html.Div(id = 'upload_sc_ind', style={'color': 'yellow', 'padding-top': 10}) 
                                               ]
                                    ),
                   'ccar_2019':  html.Div( children = [dcc.RadioItems(
                                                                    id = 'ccar_scenario',
                                                                    options=[
                                                                            {'label': 'Base', 'value': 'base'},
                                                                            {'label': 'Adverse', 'value': 'adverse'},
                                                                            {'label': 'Severely Adverse', 'value': 'severely'}
                                                                            ],
                                                                    value='base',
                                                                    labelStyle ={'color': 'white', 'display': 'inline-block'}
                                                                    ),
                                                     html.Div(id = 'ccar_sc_ind', style={'color': 'yellow', 'padding-top': 10})
                                                     ]
                                         )                   
                  }

@app.callback(Output('sc_input_container','children'), [Input('sc_input_method', 'value')])
def final_model_print2(sc):
    return scenario_input[sc]  

@app.callback([Output('upload_scenario_store', 'children'), Output('upload_sc_ind', 'children')],
              [Input('upload_scenario', 'contents')],
              [State('upload_scenario', 'filename')])
def scenario1(c, n):
    data = cecl_ia.parse_contents(c, n) 
    
    if data.empty:
        return data.to_json(), 'Upload not successful......'
    else:    
        return data.to_json(), 'Scenario file [{}] upload successful......'.format(n)
    
@app.callback([Output('ccar_scenario_store', 'children'), Output('ccar_sc_ind', 'children')],
             [Input('ccar_scenario', 'value')])
def scenario2(sc):
    if sc == 'base':
        ccar_sc = pd.read_csv("base.csv")
    elif sc == 'adverse':
        ccar_sc = pd.read_csv("adv.csv")
    elif sc == 'severely':
        ccar_sc = pd.read_csv("sa.csv")
    return ccar_sc.to_json(), 'CCAR 2019 {} scenario is chosen'.format(sc)

@app.callback(Output('rs_slider_output', 'children'),
              [Input('rs_len', 'value')])
def rs_update_output(rs_len):
    return 'Reasonable and Supportable Period (Qtr): {}'.format(rs_len)

@app.callback([Output('PP_val', 'style'), Output('pp_slider_output', 'style')],
              [Input('PP_option', 'value')])
def pp_slider_hide(PP_option):
    if PP_option == 'pp_data':
        return {'display': 'none'}, {'display': 'none'}
    elif PP_option == 'pp_user':
        return {'display':'inline-block', 'width':'100%'}, {'padding-top':5}
    
@app.callback(Output('pp_slider_output', 'children'),
              [Input('PP_val', 'value')])
def pp_update_output(PP_val):
    return 'Choose Prepayment Rate: {}'.format(PP_val)
     
@app.callback([Output('LGD_val', 'style'), Output('lgd_slider_output', 'style')],
              [Input('LGD_option', 'value')])
def lgd_slider_hide(LGD_option):
    if LGD_option == 'lgd_data':
        return {'display': 'none'}, {'display': 'none'}
    elif LGD_option == 'lgd_fj':
        return {'display': 'none'}, {'display': 'none'}
    elif LGD_option == 'lgd_user':
        return {'display':'inline-block', 'width':'100%'}, {'padding-top':5}
    
@app.callback(Output('lgd_slider_output', 'children'),
              [Input('LGD_val', 'value')])
def lgd_update_output(LGD_val):
    return 'Choose Value for Fixed LGD: {}'.format(LGD_val)



    
#@app.callback(Output('LGD_val', 'style'),
#              [Input('LGD_option', 'value')])
#def lgd_slider_hide(LGD_option):
#    if LGD_option == 'lgd_data':
#        return {'display': 'none'}
#    elif LGD_option == 'lgd_fj':
#        return {'display': 'none'}
#    elif LGD_option == 'lgd_user':
#        return {'display':'inline-block', 'width':370}
@app.callback([Output('Port_plot','figure'), Output('Scenario_plot','figure')],
              [Input('btn1', 'n_clicks')],
              [State('upload_portfolio_store', 'children'), State('sc_input_method', 'value'),
               State('upload_scenario_store', 'children'), State('ccar_scenario_store', 'children')])
def port_plot(n_clicks, port, sc_input, up_sc, ccar_sc):
    port_data = pd.read_json(port)
    port_data = port_data.sort_values('Group.1')
    fig1 = cecl_ia.port_plot(port_data)
    
    if sc_input == 'upload':
        scenario = pd.read_json(up_sc)
        scenario = scenario.sort_values('Date')
    elif sc_input == 'ccar_2019':
        scenario = pd.read_json(ccar_sc)
        scenario = scenario.sort_values('Date')
        
    fig2 = cecl_ia.sc_plot(scenario)
    
    return fig1, fig2

@app.callback([Output('ecl_frct_store','children'), Output('ecl_frct_ind', 'children'),
               Output('pd_frct_store','children'), Output('lgd_frct_store','children')],
              [Input('btn2','n_clicks')],
              [State('upload_portfolio_store', 'children'), State('sc_input_method', 'value'),
               State('upload_scenario_store', 'children'), State('ccar_scenario_store', 'children'),
               State('reversion', 'value'), State('rs_len', 'value'),
               State('LGD_option', 'value'), State('LGD_val', 'value'),
               State('PP_option', 'value'), State('PP_val', 'value'),
               State('EAD_option', 'value')])
def ecl_fcst(n_clicks, up_port, sc_input, up_sc, ccar_sc, reversion, rs_len, LGD_option, LGD_val, PP_option, PP_val, EAD_option):
    
    start_time = time.time()
    portfolio = pd.read_json(up_port)
    portfolio = portfolio.sort_values('Group.1')

    if sc_input == 'upload':
        scenario = pd.read_json(up_sc)
        scenario = scenario.sort_values('Date')
    elif sc_input == 'ccar_2019':
        scenario = pd.read_json(ccar_sc)
        scenario = scenario.sort_values('Date')
    
    ecl_loan, ecl_pd, ecl_lgd = cecl_ia.ECL_Calc(portfolio, scenario, reversion, rs_len, LGD_option, LGD_val, PP_option, PP_val, EAD_option) 
    
    run_time = '--- run time is {}s seconds ---'.format(round(time.time() - start_time, 2))
    
    return ecl_loan.to_json(), run_time, ecl_pd.to_json(), ecl_lgd.to_json()


@app.callback(Output('ECL_plot','figure'),
              [Input('ecl_frct_store', 'children')])
def ecl_plot(ecl):
    data = pd.read_json(ecl)
    data = data.sort_values('Group.1')
    fig = cecl_ia.ecl_plot(data)
    
    return fig

@app.callback(Output('PD_LGD_plot','figure'),
              [Input('pd_frct_store', 'children'), Input('lgd_frct_store', 'children'),
               Input('pd_plot_option', 'value')])
def pd_lgd_plot(data1, data2, group):
    pd_data  = pd.read_json(data1)
    pd_data  = pd_data.sort_values('obs')
    lgd_data = pd.read_json(data2)
    lgd_data = lgd_data.sort_values('obs')
    fig = cecl_ia.pd_lgd_plot(pd_data, lgd_data, group)
    
    return fig

if __name__ == '__main__':
    app.run_server()