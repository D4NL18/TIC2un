import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class NeurofuzzyService {

  private apiUrl = 'http://localhost:5000/nf';

  constructor(private http: HttpClient) { }

  trainNF(): Observable<any> {
    return this.http.post(`${this.apiUrl}/run`, {});
  }

  getImage(): Observable<any> {
    return this.http.get(`${this.apiUrl}/image`, { responseType: 'blob' });
  }

}